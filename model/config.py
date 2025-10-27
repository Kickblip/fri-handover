"""
Central configuration for the handover transformer.
All outputs live under dataset/model_output/ as requested.
"""
from pathlib import Path

# ---------------- Input roots (match your repo layout) ----------------
ROOT = Path("dataset")
RODR_DIR       = ROOT / "mediapipe_outputs" / "csv" / "rodrigues"            # {stem}_rodrigues.csv
VERTICES_DIR   = ROOT / "vertices_csv"                                       # {stem}_vertices.csv
GLOBAL_MIN_DIR = ROOT / "mediapipe_outputs" / "csv" / "closest_pair_distance" # {stem}_closest_global.csv

# ---------------- Output roots (under dataset/) ----------------
MODEL_OUT_DIR = ROOT / "model_output"
CKPT_DIR      = MODEL_OUT_DIR / "checkpoints"
PRED_DIR      = MODEL_OUT_DIR / "predictions"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

# File names
CKPT_PATH = CKPT_DIR / "handover_transformer.pt"   # trained model file

# ---------------- Windowing & batching ----------------
SEQ_LEN    = 30       # frames per window (~1s @ 30 FPS)
SEQ_STRIDE = 5        # overlap for more training windows
BATCH_SIZE = 64

# ---------------- Model size ----------------
D_MODEL  = 256
N_HEAD   = 8
N_LAYERS = 4
FFN_DIM  = 512
DROPOUT  = 0.10

# ---------------- Optimization ----------------
LR                  = 2e-4
MAX_EPOCHS          = 40
POS_CLASS_WEIGHT    = 5.0     # positives are rare; upweight
EARLY_STOP_PATIENCE = 5       # stop if val F1 stalls
