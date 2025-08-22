import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
from configs.paths import get_checkpoint_path, CHECKPOINT_DIR


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args, start_task=0):

    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(args["model_name"], args["dataset"], 
        init_cls, args["increment"], args["prefix"], args["seed"],args["convnet_type"],)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random()
    _set_device(args)
    print_args(args)
    data_manager = DataManager(args["dataset"],args["shuffle"],args["seed"],args["init_cls"],args["increment"], )
    model = factory.get_model(args["model_name"], args)
    model.save_dir=logs_name

    # بارگذاری چک‌پوینت اگر start_task بزرگتر از 0 باشد
    if start_task > 0:
        checkpoint_path = get_checkpoint_path(start_task - 1)  # چون بعد از تسک ذخیره می‌شود
        if os.path.exists(checkpoint_path):
            logging.info(f"در حال بارگذاری چک‌پوینت از تسک {start_task - 1}")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=args["device"][0])
                model.load_state_dict(checkpoint['model_state_dict'])
                # بارگذاری سایر stateها اگر وجود دارند
                if 'global_prototypes' in checkpoint:
                    model.global_prototypes = checkpoint['global_prototypes'].to(args["device"][0])
                if 'prototype_memory' in checkpoint:
                    model.prototype_memory = checkpoint['prototype_memory']
                if 'known_classes' in checkpoint:
                    model._known_classes = checkpoint['known_classes']
                logging.info(f"با موفقیت از تسک {start_task - 1} ادامه داده شد")
            except Exception as e:
                logging.error(f"خطا در بارگذاری چک‌پوینت: {e}")
                logging.info("شروع آموزش از ابتدا")
                start_task = 0
        else:
            logging.warning(f"چک‌پوینت برای تسک {start_task - 1} یافت نشد")
            start_task = 0

    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    zs_seen_curve, zs_unseen_curve, zs_harmonic_curve, zs_total_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}, {"top1": [], "top5": []}, {"top1": [], "top5": []}
    logging.info("data_manager.nb_tasks=> {}".format(data_manager.nb_tasks))
    for task in range(data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        model.incremental_train(data_manager)
        
        # ذخیره چک‌پوینت پس از هر تسک
        try:
            checkpoint_data = {
                'model_state_dict': model.state_dict(),
                'global_prototypes': model.global_prototypes.cpu() if hasattr(model, 'global_prototypes') and model.global_prototypes is not None else None,
                'prototype_memory': model.prototype_memory if hasattr(model, 'prototype_memory') else {},
                'known_classes': model._known_classes if hasattr(model, '_known_classes') else 0,
                'task': task
            }
            checkpoint_path = get_checkpoint_path(task)
            torch.save(checkpoint_data, checkpoint_path)
            logging.info(f"چک‌پوینت برای تسک {task} ذخیره شد")
        except Exception as e:
            logging.error(f"خطا در ذخیره چک‌پوینت: {e}")

        cnn_accy, nme_accy, zs_seen, zs_unseen, zs_harmonic, zs_total = model.eval_task()
        model.after_task()

       
        logging.info("CNN: {}".format(cnn_accy["grouped"]))

        cnn_curve["top1"].append(cnn_accy["top1"])
        cnn_curve["top5"].append(cnn_accy["top5"])

        logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
        logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))

        print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
        logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
    logging.info("finished!!!!")
    
def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
