# utils/checkpoint_utils.py
import os
import torch
from configs.paths import get_checkpoint_path

def save_checkpoint(model, task_id, additional_data=None):
    """ذخیره checkpoint"""
    checkpoint_path = get_checkpoint_path(task_id)
    
    checkpoint_data = {
        'model_state_dict': model.state_dict(),
        'task_id': task_id,
    }
    
    if additional_data:
        checkpoint_data.update(additional_data)
    
    torch.save(checkpoint_data, checkpoint_path)
    print(f"Checkpoint saved for task {task_id} at {checkpoint_path}")

def load_checkpoint(model, task_id, device='cuda'):
    """بارگذاری checkpoint"""
    checkpoint_path = get_checkpoint_path(task_id)
    
    if not os.path.exists(checkpoint_path):
        return None
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Checkpoint loaded from task {task_id}")
    return checkpoint