# FRI-HANDOVER Deep Learning System Architecture

## Overview

The FRI-HANDOVER system is a **Transformer-based neural network** that predicts future hand trajectories during handover events. It consists of data collection, preprocessing, model training, and inference components.

---

## Complete System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA COLLECTION PIPELINE                             │
└─────────────────────────────────────────────────────────────────────────────┘

    Azure Kinect DK
    ┌──────────────┐
    │  RGB Video   │ ──┐
    │  Depth Data  │ ──┼──> MKV Video File
    │  Calibration │ ──┘
    └──────────────┘

                    ┌─────────────────────────────────┐
                    │   Processing Pipeline            │
                    │   (record/pipeline/run.py)       │
                    └─────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
        ┌───────▼────────┐         ┌────────▼────────┐
        │  MediaPipe     │         │  AprilTag       │
        │  Hand Tracking │         │  Detection      │
        └───────┬────────┘         └────────┬────────┘
                │                           │
                │ 21 landmarks × 2 hands    │ 8 vertices × 3 coords
                │ (x, y, z) per landmark    │ (x, y, z) per vertex
                │                           │
        ┌───────▼───────────────────────────▼────────┐
        │  CSV Files: {stem}_hands.csv                │
        │          {stem}_box.csv                    │
        │  Format: frame_idx, h0_lm0_x, h0_lm0_y,   │
        │          h0_lm0_z, ..., h1_lm0_x, ...,     │
        │          v0_x, v0_y, v0_z, ...             │
        └────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA PREPROCESSING                                  │
└─────────────────────────────────────────────────────────────────────────────┘

    CSV Files
    ┌──────────────┐
    │ {stem}_hands │ ──┐
    │ {stem}_box   │ ──┼──> load_features()
    └──────────────┘   │
                       │
    ┌──────────────────▼──────────────────┐
    │  Feature Concatenation               │
    │  ┌────────────────────────────────┐ │
    │  │ Hand 0: 63 features            │ │
    │  │ (21 landmarks × 3 coords)      │ │
    │  ├────────────────────────────────┤ │
    │  │ Hand 1: 63 features            │ │
    │  │ (21 landmarks × 3 coords)      │ │
    │  ├────────────────────────────────┤ │
    │  │ Box: 24 features              │ │
    │  │ (8 vertices × 3 coords)       │ │
    │  └────────────────────────────────┘ │
    │  Total: 150 features per frame    │
    └──────────────────┬──────────────────┘
                       │
    ┌──────────────────▼──────────────────┐
    │  Sliding Window Sequence Generation │
    │  ┌────────────────────────────────┐ │
    │  │ Input:  [10 frames × 150 feat] │ │
    │  │ Target: [20 frames × 63 feat]  │ │
    │  │         (receiving hand only)  │ │
    │  └────────────────────────────────┘ │
    └──────────────────┬──────────────────┘
                       │
              ┌────────▼────────┐
              │  PyTorch Dataset│
              │  DataLoader     │
              └────────┬────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                    TRANSFORMER MODEL ARCHITECTURE                            │
└─────────────────────────────────────────────────────────────────────────────┘

    Input Batch
    ┌─────────────────────────────────────┐
    │ [B, 10, 150]                       │
    │ B = batch size (64)                │
    │ 10 = sequence length (frames)      │
    │ 150 = features (hands + box)       │
    └────────────┬────────────────────────┘
                 │
    ┌────────────▼────────────────────────┐
    │  Input Projection                  │
    │  Linear(150 → 512)                 │
    │  [B, 10, 150] → [B, 10, 512]       │
    └────────────┬────────────────────────┘
                 │
    ┌────────────▼────────────────────────┐
    │  Positional Encoding                │
    │  Adds temporal position info       │
    │  Frame 0, Frame 1, ..., Frame 9    │
    │  [B, 10, 512]                       │
    └────────────┬────────────────────────┘
                 │
    ┌────────────▼────────────────────────┐
    │  TRANSFORMER ENCODER                │
    │  ┌────────────────────────────────┐ │
    │  │ Layer 1:                        │ │
    │  │  ┌──────────────────────────┐  │ │
    │  │  │ Multi-Head Self-Attention│  │ │
    │  │  │ (8 heads, 512 dim)       │  │ │
    │  │  │                          │  │ │
    │  │  │ Each frame attends to    │  │ │
    │  │  │ ALL other frames         │  │ │
    │  │  └──────────────────────────┘  │ │
    │  │  ┌──────────────────────────┐  │ │
    │  │  │ Feed-Forward Network     │  │ │
    │  │  │ (512 → 512 → 512)        │  │ │
    │  │  └──────────────────────────┘  │ │
    │  └────────────────────────────────┘ │
    │  (Repeated 4 times)                 │
    │  └────────────────────────────────┘ │
    │  Output: [B, 10, 512]               │
    │  (Temporal relationships learned)    │
    └────────────┬────────────────────────┘
                 │
    ┌────────────▼────────────────────────┐
    │  Memory (Encoder Output)           │
    │  [B, 10, 512]                       │
    └────────────┬────────────────────────┘
                 │
    ┌────────────▼────────────────────────┐
    │  TRANSFORMER DECODER                 │
    │  ┌────────────────────────────────┐ │
    │  │ Learnable Query Embeddings    │ │
    │  │ [B, 20, 512]                   │ │
    │  │ (20 future frames to predict) │ │
    │  └────────────────────────────────┘ │
    │  ┌────────────────────────────────┐ │
    │  │ Layer 1:                        │ │
    │  │  ┌──────────────────────────┐  │ │
    │  │  │ Cross-Attention           │  │ │
    │  │  │ (Queries attend to Memory)│  │ │
    │  │  └──────────────────────────┘  │ │
    │  │  ┌──────────────────────────┐  │ │
    │  │  │ Self-Attention            │  │ │
    │  │  │ (Future frames attend to  │  │ │
    │  │  │  each other)              │  │ │
    │  │  └──────────────────────────┘  │ │
    │  │  ┌──────────────────────────┐  │ │
    │  │  │ Feed-Forward Network     │  │ │
    │  │  └──────────────────────────┘  │ │
    │  └────────────────────────────────┘ │
    │  (Repeated 4 times)                 │
    │  Output: [B, 20, 512]               │
    └────────────┬────────────────────────┘
                 │
    ┌────────────▼────────────────────────┐
    │  Output Projection                 │
    │  Linear(512 → 63)                  │
    │  [B, 20, 512] → [B, 20, 63]        │
    │  63 = 21 landmarks × 3 coords      │
    └────────────┬────────────────────────┘
                 │
    ┌────────────▼────────────────────────┐
    │  Predicted Future Frames            │
    │  [B, 20, 63]                        │
    │  20 frames of receiving hand        │
    │  world coordinates                  │
    └─────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PROCESS                                    │
└─────────────────────────────────────────────────────────────────────────────┘

    Training Data
    ┌─────────────────────────────────────┐
    │ Input:  [B, 10, 150]               │
    │ Target: [B, 20, 63]                 │
    └────────────┬────────────────────────┘
                 │
    ┌────────────▼────────────────────────┐
    │  Forward Pass                       │
    │  Transformer → Predictions           │
    │  [B, 20, 63]                        │
    └────────────┬────────────────────────┘
                 │
    ┌────────────▼────────────────────────┐
    │  Loss Calculation                   │
    │  MSE(Predictions, Targets)         │
    │  Mean Squared Error                │
    └────────────┬────────────────────────┘
                 │
    ┌────────────▼────────────────────────┐
    │  Backpropagation                   │
    │  Update model weights              │
    │  Optimizer: AdamW (lr=2e-4)        │
    └────────────┬────────────────────────┘
                 │
    ┌────────────▼────────────────────────┐
    │  Checkpoint Saving                  │
    │  Best model → handover_transformer.pt│
    └─────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                         INFERENCE PROCESS                                    │
└─────────────────────────────────────────────────────────────────────────────┘

    New Video
    ┌─────────────────────────────────────┐
    │ Process through pipeline           │
    │ → CSV files with hand/box data     │
    └────────────┬────────────────────────┘
                 │
    ┌────────────▼────────────────────────┐
    │  Load Trained Model                │
    │  handover_transformer.pt           │
    └────────────┬────────────────────────┘
                 │
    ┌────────────▼────────────────────────┐
    │  Sliding Window Prediction          │
    │  Frames 0-9   → Predict 10-29       │
    │  Frames 20-29 → Predict 30-49      │
    │  Frames 40-49 → Predict 50-69      │
    │  ...                                 │
    └────────────┬────────────────────────┘
                 │
    ┌────────────▼────────────────────────┐
    │  Output                             │
    │  ┌────────────────────────────────┐ │
    │  │ CSV: Predictions               │ │
    │  │ Video: Visualization overlay   │ │
    │  └────────────────────────────────┘ │
    └─────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                    KEY COMPONENTS SUMMARY                                   │
└─────────────────────────────────────────────────────────────────────────────┘

1. DATA COLLECTION
   ├─ Azure Kinect DK (RGB-D sensor)
   ├─ MediaPipe (hand tracking)
   └─ AprilTag (object tracking)

2. DATA PREPROCESSING
   ├─ Feature extraction (hands + box)
   ├─ Hand consistency algorithm
   └─ Sequence generation (sliding windows)

3. DEEP LEARNING MODEL
   ├─ Transformer Encoder (temporal modeling)
   ├─ Transformer Decoder (future prediction)
   └─ Multi-head self-attention (8 heads)

4. TRAINING
   ├─ Loss: Mean Squared Error (MSE)
   ├─ Optimizer: AdamW
   └─ Early stopping on validation loss

5. INFERENCE
   ├─ Load trained checkpoint
   ├─ Predict future frames
   └─ Generate visualization


┌─────────────────────────────────────────────────────────────────────────────┐
│                    MODEL HYPERPARAMETERS                                    │
└─────────────────────────────────────────────────────────────────────────────┘

Architecture:
  - d_model: 512 (model dimension)
  - n_heads: 8 (attention heads)
  - n_layers: 4 (encoder + decoder layers)
  - ffn_dim: 512 (feed-forward dimension)
  - dropout: 0.10

Sequence:
  - seq_len: 10 frames (~0.33s @ 30fps)
  - future_frames: 20 frames (~0.67s ahead)
  - stride: 5 frames (50% overlap)

Training:
  - batch_size: 64
  - learning_rate: 2e-4
  - max_epochs: 40
  - early_stop_patience: 5

Input/Output:
  - Input: 150 features (126 hands + 24 box)
  - Output: 63 features (21 landmarks × 3 coords)


┌─────────────────────────────────────────────────────────────────────────────┐
│                    DATA FLOW EXAMPLE                                        │
└─────────────────────────────────────────────────────────────────────────────┘

Frame-by-Frame Processing:

Frame 0: [hand0_coords(63), hand1_coords(63), box_coords(24)] = 150 features
Frame 1: [hand0_coords(63), hand1_coords(63), box_coords(24)] = 150 features
Frame 2: [hand0_coords(63), hand1_coords(63), box_coords(24)] = 150 features
...
Frame 9: [hand0_coords(63), hand1_coords(63), box_coords(24)] = 150 features
         └─────────────────────────────────────────────────────┘
                          Input Sequence [10 × 150]

         ┌─────────────────────────────────────────────────────┐
         │  Transformer Encoder                                 │
         │  Self-Attention learns relationships between frames  │
         └─────────────────────────────────────────────────────┘

         ┌─────────────────────────────────────────────────────┐
         │  Transformer Decoder                                 │
         │  Predicts future receiving hand positions            │
         └─────────────────────────────────────────────────────┘

Predicted Frame 10: [receiving_hand_coords(63)]
Predicted Frame 11: [receiving_hand_coords(63)]
...
Predicted Frame 29: [receiving_hand_coords(63)]
         └─────────────────────────────────────────────────────┘
                    Output Sequence [20 × 63]

