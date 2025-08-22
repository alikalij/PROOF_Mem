import argparse
import json
import os
import torch
from trainer import train, _train
from configs.paths import get_checkpoint_path, CHECKPOINT_DIR

def find_latest_checkpoint():
    """پیدا کردن آخرین checkpoint ذخیره شده"""
    if not os.path.exists(CHECKPOINT_DIR):
        return None
        
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) 
                  if f.startswith('checkpoint_task_') and f.endswith('.pth')]
    
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
    
    return max(task_numbers) if task_numbers else None

def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json
    
    # بررسی آیا باید از checkpoint ادامه داد
    start_task = 0
    if args.get('resume', False):
        latest_task = find_latest_checkpoint()
        if latest_task is not None:
            print(f"Resuming from task {latest_task + 1}")
            start_task = latest_task + 1  # از تسک بعدی شروع کن
        else:
            print("No checkpoint found. Starting from scratch.")
    
    # اگر start_task بزرگتر از 0 است، از تابع _train با پارامتر start_task استفاده کن
    if start_task > 0:
        _train(args, start_task)
    else:
        train(args)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/test.json', help='Json file of settings.')
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    
    return parser


if __name__ == '__main__':
    main()
