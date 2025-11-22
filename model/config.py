"""
Central configuration for the handover transformer.
All outputs live under dataset/model_output/ as requested.
"""
from pathlib import Path

# ---------------- Input roots (match your repo layout) ---------------- 
MODEL_DATASET_ROOT = Path("model_dataset") / "handover-csv"
HANDS_DIR = MODEL_DATASET_ROOT / "hands"      # {number}_video_hands.csv
BOX_DIR   = MODEL_DATASET_ROOT / "box"        # {number}_video_box.csv
# Legacy paths (kept for backward compatibility if needed)
ROOT = Path("dataset")
WORLD_DIR = ROOT / "mediapipe_outputs" / "csv" / "world"  # Legacy: {stem}_world.csv
VERTICES_DIR = ROOT / "vertices_csv"  # Legacy: {stem}_vertices.csv
GLOBAL_MIN_DIR = ROOT / "mediapipe_outputs" / "csv" / "closest_pair_distance"

# ---------------- Output roots (under dataset/) ---------------- 
MODEL_OUT_DIR = ROOT / "model_output"
CKPT_DIR      = MODEL_OUT_DIR / "checkpoints"
PRED_DIR      = MODEL_OUT_DIR / "predictions"
VIDEO_DIR     = MODEL_OUT_DIR / "videos"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

# File names
CKPT_PATH = CKPT_DIR / "handover_transformer.pt"   # trained model file

# ---------------- Windowing & batching ---------------- 
SEQ_LEN    = 20       # frames per window (~0.67s @ 30 FPS)
SEQ_STRIDE = 5        # overlap for more training windows
BATCH_SIZE = 64
FUTURE_FRAMES = 10    # number of frames to predict ahead

# ---------------- Model size ---------------- 
D_MODEL  = 256
N_HEAD   = 8
N_LAYERS = 4
FFN_DIM  = 512
DROPOUT  = 0.10

# ---------------- Optimization ---------------- 
LR                  = 2e-4
MAX_EPOCHS          = 40
EARLY_STOP_PATIENCE = 5       # stop if val loss stalls
