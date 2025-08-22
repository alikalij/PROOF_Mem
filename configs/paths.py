# config/paths.py
import os
from google.colab import drive

# Mount Google Drive (فقط در Colab)
drive.mount('/content/drive')

# مسیرهای اصلی
CHECKPOINT_DIR = "/content/drive/MyDrive/saved_model/PROOF_Mem_Checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def get_checkpoint_path(task_id=None, emergency=False):
    """ایجاد مسیر فایل checkpoint"""
    if emergency:
        return os.path.join(CHECKPOINT_DIR, "emergency_checkpoint.pth")
    elif task_id is not None:
        return os.path.join(CHECKPOINT_DIR, f"checkpoint_task_{task_id}.pth")
    return CHECKPOINT_DIR