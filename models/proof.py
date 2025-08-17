import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import Proof_Net
from models.base import BaseLearner
from utils.toolkit import tensor2numpy, get_attribute, ClipLoss
import os

num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        
        self._train_transformer = False
        self._network = Proof_Net(args, False)
        
        self.batch_size = get_attribute(args, "batch_size", 48)
        self.init_lr = get_attribute(args, "init_lr", 0.01)
        self.weight_decay = get_attribute(args, "weight_decay", 0.0005)
        self.min_lr = get_attribute(args, "min_lr", 1e-8)
        self.frozen_layers = get_attribute(args, "frozen_layers", None)
        self.tuned_epoch = get_attribute(args, "tuned_epoch", 5)
        
        self._known_classes = 0
        self.use_cos = get_attribute(args, "use_cos", False)
        
        # Enhanced Prototype Memory
        self.global_prototypes = None
        self.prototype_memory = {}
        self.prototype_update_factor = 0.7  # Weight for prototype updates

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))
    
    def cal_prototype(self, trainloader, model):
        model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_, data, label) = batch
                data = data.to(self._device)
                embedding = model.convnet.encode_image(data, True)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        class_list = list(range(self._known_classes, self._total_classes))
        for class_index in class_list:
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            embedding = embedding_list[data_index]
            proto = embedding.mean(0)
            
            # Enhanced prototype memory management
            if class_index in self.prototype_memory:
                # Update existing prototype with momentum
                old_proto = self.prototype_memory[class_index].to(self._device)
                new_proto = self.prototype_update_factor * old_proto + (1 - self.prototype_update_factor) * proto
                self.prototype_memory[class_index] = new_proto.cpu()
                self._network.img_prototypes[class_index] = new_proto.to(self._device)
            else:
                # Initialize new prototype
                self.prototype_memory[class_index] = proto
                self._network.img_prototypes[class_index] = proto.to(self._device)
    
    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        
        # Initialize prototype memory
        if self.global_prototypes is None:
            self.global_prototypes = torch.zeros(
                (self._total_classes, self._network.img_prototypes.shape[1]),
                device=self._device
            )
        else:
            # Expand global prototypes for new classes
            new_prototypes = torch.zeros(
                (self._total_classes - self.global_prototypes.shape[0], 
                 self.global_prototypes.shape[1]),
                device=self._device
            )
            self.global_prototypes = torch.cat([self.global_prototypes, new_prototypes], dim=0)
        
        self._network.update_prototype(self._total_classes)
        self._network.update_context_prompt()
        self._network.extend_task()
        
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train", mode="train", appendent=self._get_memory()
        )
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self._network.to(self._device)
       
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers
        )

        train_dataset_for_protonet = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="test"
        )
        self.train_loader_for_protonet = DataLoader(
            train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self.cal_prototype(self.train_loader_for_protonet, self._network)
        
        # Update global prototypes with new knowledge
        for class_idx in range(self._known_classes, self._total_classes):
            self.global_prototypes[class_idx] = self._network.img_prototypes[class_idx].clone()
        
        self._train_proj(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train_proj(self, train_loader, test_loader):
        self._train_transformer = True
        self._network.to(self._device)
       
        # Freeze appropriate layers
        for name, param in self._network.convnet.named_parameters():
            if 'logit_scale' not in name:
                param.requires_grad = False
        self._network.freeze_projection_weight_new()
        
        # Optimizer setup
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(self._network.parameters(), momentum=0.9, 
                                 lr=self.init_lr, weight_decay=self.weight_decay)
        elif self.args['optimizer'] == 'adam': 
            optimizer = optim.AdamW(self._network.parameters(), lr=self.init_lr, 
                                   weight_decay=self.weight_decay)
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr
        )

        class_to_label = self.data_manager._class_to_label
        templates = self.data_manager._data_to_prompt[0]
        prog_bar = tqdm(range(self.tuned_epoch))
        cliploss = ClipLoss()

        total_labels = class_to_label[:self._total_classes]
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            
            for i, (_, inputs, targets) in enumerate(train_loader):
                labels = [class_to_label[y] for y in targets]
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)
                
                # Text features
                texts = [templates.format(inst) for inst in total_labels]
                texts = self._network.tokenizer(texts).to(self._device)
                text_features = self._network.encode_text(texts)
                text_feas = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Image features
                image_features = self._network.encode_image(inputs)
                img_feas = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Forward pass
                image_features, text_features, logit_scale, _ = self._network.forward_transformer(
                    img_feas, text_feas, self._train_transformer
                )
                logits = image_features @ text_features.T
                
                # Enhanced prototype loss using global memory
                proto_outputs = image_features @ self.global_prototypes.T
                protoloss = F.cross_entropy(proto_outputs, targets)
                
                # CLIP consistency loss
                clip_texts = [templates.format(inst) for inst in labels]
                clip_text_feas = self._network.encode_text(self._network.tokenizer(clip_texts).to(self._device))
                clip_text_feas = clip_text_feas / clip_text_feas.norm(dim=-1, keepdim=True)
                clip_loss = cliploss(img_feas, clip_text_feas, logit_scale)
                
                # Classification loss
                loss = F.cross_entropy(logits, targets)
                
                # Combined loss with enhanced weights
                total_loss = loss + 0.5 * clip_loss + 0.7 * protoloss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                losses += total_loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_acc {:.2f}, Test_acc {:.2f}".format(
                self._cur_task, epoch + 1, self.args['tuned_epoch'], 
                losses / len(train_loader), train_acc, test_acc
            )
            prog_bar.set_description(info)

    def _compute_accuracy(self, model, loader):
        model.eval()
        class_to_label = self.data_manager._class_to_label
        templates = self.data_manager._data_to_prompt
        total_labels = class_to_label[:self._total_classes]
        
        # Prepare text features
        text_features = []
        with torch.no_grad():
            for l in total_labels:
                texts = [t.format(l) for t in templates]
                texts = model.tokenizer(texts).to(self._device)
                class_embeddings = model.encode_text(texts)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                class_embeddings = class_embeddings.mean(dim=0)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                text_features.append(class_embeddings)
            text_features = torch.stack(text_features, dim=0)
        
        # Evaluation with enhanced prototypes
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                # Extract features
                image_features = model.encode_image(inputs)
                transf_image_features, transf_text_features, _, _ = model.forward_transformer(
                    image_features, text_features, self._train_transformer
                )
                
                # Enhanced prediction with global prototypes
                original_outputs = image_features @ text_features.T
                transformer_outputs = transf_image_features @ transf_text_features.T
                proto_outputs = transf_image_features @ self.global_prototypes.T
                
                outputs = original_outputs + transformer_outputs + proto_outputs

            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)