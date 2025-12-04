"""
Configuration for the TCN Model (Standalone).
"""
from pathlib import Path

# ---------------- Input roots ---------------- 
MODEL_DATASET_ROOT = Path("model_dataset") / "handover-csv" / "handover-csv"
HANDS_DIR = MODEL_DATASET_ROOT / "hands"
BOX_DIR   = MODEL_DATASET_ROOT / "box"

# Legacy paths
ROOT = Path("dataset")
WORLD_DIR = ROOT / "mediapipe_outputs" / "csv" / "world"
VERTICES_DIR = ROOT / "vertices_csv"

# ---------------- Output roots ---------------- 
MODEL_OUT_DIR = ROOT / "model_output"
CKPT_DIR      = MODEL_OUT_DIR / "checkpoints"
PRED_DIR      = MODEL_OUT_DIR / "predictions"
VIDEO_DIR     = MODEL_OUT_DIR / "videos"

CKPT_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- TCN Specifics ---------------- 
# Using 256 as hidden size (similar capacity to D_MODEL)
TCN_HIDDEN_CHANNELS = [1024, 1024, 1024, 1024]              
TCN_KERNEL_SIZE = 3
TCN_DROPOUT = 0.3

# ---------------- Paramaters---------------- 
SEQ_LEN    = 20        
SEQ_STRIDE = 1
BATCH_SIZE = 32
FUTURE_FRAMES = 10     
# ---------------- Data Splitting ----------------
TRAIN_SPLIT = 0.7
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.15
# ---------------- Optimization ---------------- 
LR = 2e-4
MAX_EPOCHS = 40
EARLY_STOP_PATIENCE = 5