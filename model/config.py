"""
Configuration file for the Handover Transformer.

This defines paths, model hyperparameters, and training settings.
"""

from pathlib import Path

# ---- Directory setup ----
ROOT = Path("dataset")  # base dataset folder
RODR_DIR = ROOT / "mediapipe_outputs" / "csv" / "rodrigues"            # per-hand Rodrigues CSVs
GLOBAL_MIN_DIR = ROOT / "mediapipe_outputs" / "csv" / "closest_pair_distance"  # global min distance CSVs

# ---- Data parameters ----
SEQ_LEN = 30       # frames per input sequence (~1 second)
SEQ_STRIDE = 5     # how much we slide the window each time (overlap between sequences)
BATCH_SIZE = 64    # samples per gradient update

# ---- Model parameters ----
D_MODEL = 256      # embedding dimension
N_HEAD = 8         # number of attention heads
N_LAYERS = 4       # number of transformer encoder layers
FFN_DIM = 512      # dimension of feedforward layer inside transformer
DROPOUT = 0.1      # dropout rate

# ---- Training parameters ----
LR = 2e-4                # learning rate
MAX_EPOCHS = 40          # max training epochs
POS_CLASS_WEIGHT = 5.0   # weight to handle imbalance (handover frames are rare)
EARLY_STOP_PATIENCE = 5  # stop training if no improvement after N epochs
