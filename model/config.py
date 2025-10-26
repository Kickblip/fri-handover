"""
Configuration for the Handover Transformer.

Central place to adjust:
- paths to your dataset artifacts,
- sequence/windowing parameters,
- model architecture sizes,
- training hyperparameters.
"""

from pathlib import Path

# ---------------- Paths (match your pipeline output layout) ----------------
ROOT = Path("dataset")

# Model INPUT features:
#   - Rodrigues: per-frame orientation representation for two hands (concatenated)
#   - Vertices : per-frame object geometry (e.g., baton vertices from AprilTags or other estimator)
RODR_DIR      = ROOT / "mediapipe_outputs" / "csv" / "rodrigues"           # {stem}_rodrigues.csv
VERTICES_DIR  = ROOT / "vertices_csv"                                       # {stem}_vertices.csv

# Proxy LABELS for training (weak supervision):
#   - closest_global: a CSV with the frame index at which inter-hand distance is minimal (argmin).
#     We turn ±5 frames around that into "handover=1" and the rest into 0.
GLOBAL_MIN_DIR = ROOT / "mediapipe_outputs" / "csv" / "closest_pair_distance"  # {stem}_closest_global.csv

# ---------------- Data/windowing parameters ----------------
SEQ_LEN    = 30      # number of frames the model sees at once (~1s at 30 FPS)
SEQ_STRIDE = 5       # stride for training windows (more overlap => more samples)
BATCH_SIZE = 64      # adjust for available memory

# ---------------- Model size ----------------
D_MODEL  = 256       # width of the transformer embeddings
N_HEAD   = 8         # multi-head attention heads
N_LAYERS = 4         # number of encoder layers
FFN_DIM  = 512       # inner MLP width inside each transformer block
DROPOUT  = 0.10      # dropout to regularize

# ---------------- Optimization ----------------
LR                  = 2e-4    # AdamW learning rate
MAX_EPOCHS          = 40      # upper bound on epochs
POS_CLASS_WEIGHT    = 5.0     # upweight positives (handover frames are rare)
EARLY_STOP_PATIENCE = 5       # stop if val F1 doesn't improve for N epochs
