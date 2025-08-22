#!/usr/bin/env python3
"""
اسکریپت آموزش قابلیت ادامه‌دادن از آخرین checkpoint با پشتیبانی از آرگومان‌های خط فرمان
"""

import os
import argparse
import torch
import json
from trainer import train
from models.proof import Learner
from utils.data_manager import DataManager
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

CHECKPOINT_DIR = "/content/drive/MyDrive/saved_model/PROOF_Mem_Checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def parse_args():
    """پردازش آرگومان‌های خط فرمان"""
    parser = argparse.ArgumentParser(description='Resumable PROOF Training')
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to config file')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from latest checkpoint')
    return parser.parse_args()

def load_config(config_path):
    """لود کردن فایل کانفیگ"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def find_latest_checkpoint():
    """پیدا کردن آخرین checkpoint ذخیره شده"""
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith('checkpoint_task_')]
    if not checkpoints:
        return None
    
    # استخراج شماره تسک از نام فایل
    task_numbers = []
    for checkpoint in checkpoints:
        try:
            task_num = int(checkpoint.split('_')[2].split('.')[0])
            task_numbers.append(task_num)
        except:
            continue
    
    if not task_numbers:
        return None
    
    latest_task = max(task_numbers)
    return latest_task

def main():
    # پردازش آرگومان‌های خط فرمان
    args = parse_args()
    
    # لود کردن کانفیگ
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")
    
    # پیدا کردن آخرین checkpoint
    latest_task = find_latest_checkpoint() if args.resume else None
    
    if latest_task is not None:
        print(f"Found checkpoint for task {latest_task}. Resuming training...")
        
        # ایجاد دیتا منیجر
        data_manager = DataManager(
            config['dataset'], config['shuffle'], config['seed'],
            config['init_cls'], config['increment']
        )
        
        # ایجاد مدل و بارگذاری checkpoint
        learner = Learner(config)
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_task_{latest_task}.pth")
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=config['device'][0])
            learner._network.load_state_dict(checkpoint['model_state_dict'])
            learner.global_prototypes = checkpoint['global_prototypes']
            learner.prototype_memory = checkpoint['prototype_memory']
            learner._known_classes = checkpoint['known_classes']
            
            print(f"Successfully loaded checkpoint for task {latest_task}")
            
            # ادامه آموزش از تسک بعدی
            for task in range(latest_task + 1, config['nb_tasks']):
                learner.incremental_train(data_manager)
        else:
            print("Checkpoint file not found. Starting from scratch...")
            train(config)
    else:
        print("No checkpoint found or resume flag not set. Starting from scratch...")
        train(config)

if __name__ == "__main__":
    main()