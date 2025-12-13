# FRI-HANDOVER Deep Learning System - Simplified Diagram

## What is the Deep Learning System?

The deep learning system is a **Transformer neural network** that learns to predict future hand positions by analyzing sequences of hand and object motion data.

---

## Core Deep Learning Model

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT LAYER                                  │
│  [Batch, 10 frames, 150 features]                              │
│                                                                  │
│  Features per frame:                                             │
│  ├─ Hand 0 (receiving): 63 features (21 landmarks × 3 coords)  │
│  ├─ Hand 1 (giving):    63 features (21 landmarks × 3 coords)  │
│  └─ Box (object):       24 features (8 vertices × 3 coords)     │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              INPUT PROJECTION + POSITIONAL ENCODING            │
│  Linear(150 → 512) + Add temporal position info                │
│  Output: [Batch, 10, 512]                                      │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TRANSFORMER ENCODER                          │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Layer 1: Multi-Head Self-Attention (8 heads)          │  │
│  │  ┌──────────────────────────────────────────────────┐ │  │
│  │  │  Frame 0 ◄───► Frame 1 ◄───► Frame 2 ◄───► ...   │ │  │
│  │  │    ▲              ▲              ▲                │ │  │
│  │  │    └──────────────┴──────────────┘                │ │  │
│  │  │  Each frame attends to ALL other frames            │ │  │
│  │  └──────────────────────────────────────────────────┘ │  │
│  │  Layer 2: Feed-Forward Network (512 → 512 → 512)     │  │
│  └────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Layer 3: Multi-Head Self-Attention                    │  │
│  │  Layer 4: Feed-Forward Network                         │  │
│  └────────────────────────────────────────────────────────┘  │
│  (Repeated 4 times total)                                    │
│  Output: [Batch, 10, 512] - Encoded sequence with temporal  │
│          relationships learned                                 │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MEMORY (Encoder Output)                      │
│  [Batch, 10, 512] - Rich representations of past frames        │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TRANSFORMER DECODER                          │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Learnable Query Embeddings: [Batch, 20, 512]          │  │
│  │  (20 future frames to predict)                          │  │
│  └────────────────────────────────────────────────────────┘  │
│                            │                                    │
│  ┌────────────────────────▼────────────────────────┐          │
│  │  Layer 1: Cross-Attention                        │          │
│  │  ┌────────────────────────────────────────────┐ │          │
│  │  │  Future Frame Queries attend to Memory    │ │          │
│  │  │  (past encoded frames)                     │ │          │
│  │  │                                            │ │          │
│  │  │  Query (Future) ──► Memory (Past)         │ │          │
│  │  │  Frame 10 ──► [Frame 0, 1, 2, ..., 9]     │ │          │
│  │  │  Frame 11 ──► [Frame 0, 1, 2, ..., 9]     │ │          │
│  │  │  ...                                        │ │          │
│  │  │  Frame 29 ──► [Frame 0, 1, 2, ..., 9]     │ │          │
│  │  └────────────────────────────────────────────┘ │          │
│  └─────────────────────────────────────────────────┘          │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Layer 2: Self-Attention (future frames attend to each) │  │
│  │  Layer 3: Feed-Forward Network                          │  │
│  └────────────────────────────────────────────────────────┘  │
│  (Repeated 4 times total)                                     │
│  Output: [Batch, 20, 512]                                     │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT PROJECTION                             │
│  Linear(512 → 63)                                               │
│  Output: [Batch, 20, 63]                                        │
│  63 = 21 landmarks × 3 coordinates (x, y, z)                  │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PREDICTED FUTURE FRAMES                      │
│  [Batch, 20 frames, 63 features]                               │
│                                                                  │
│  Predicted receiving hand positions for next 20 frames          │
│  (~0.67 seconds ahead at 30 FPS)                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## How Self-Attention Works (Temporal Modeling)

```
Input Sequence (10 frames):
┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐
│Frame │  │Frame │  │Frame │  │Frame │  │Frame │
│  0   │  │  1   │  │  2   │  │ ...  │  │  9   │
└───┬──┘  └───┬──┘  └───┬──┘  └───┬──┘  └───┬──┘
    │         │         │         │         │
    └─────────┴─────────┴─────────┴─────────┘
              │
              ▼
    ┌─────────────────────┐
    │  Self-Attention      │
    │  Mechanism          │
    └─────────────────────┘
              │
    Each frame can "look at" and learn from ALL other frames
              │
    ┌─────────┴─────────┐
    │                    │
    ▼                    ▼
Frame 0 learns from:  Frame 5 learns from:
├─ Frame 1            ├─ Frame 0
├─ Frame 2            ├─ Frame 1
├─ Frame 3            ├─ Frame 2
├─ ...                 ├─ Frame 3
└─ Frame 9            ├─ Frame 4
                      ├─ Frame 6
                      ├─ Frame 7
                      ├─ Frame 8
                      └─ Frame 9

This allows the model to learn:
- Long-range dependencies (Frame 0 → Frame 9)
- Temporal patterns (how motion evolves over time)
- Coordinated motion (how hands move together)
- Object-hand relationships (how box position affects hands)
```

---

## Complete System Flow

```
┌──────────────┐
│  Video Data │
│  (RGB-D)    │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│  MediaPipe +     │
│  AprilTag        │
│  Processing      │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  CSV Files       │
│  (Hand + Box)    │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐      ┌──────────────────┐
│  Data            │      │  Training        │
│  Preprocessing   │─────►│  Loop            │
│  (Sequences)     │      │  (MSE Loss)      │
└──────────────────┘      └────────┬─────────┘
                                   │
                                   ▼
                          ┌──────────────────┐
                          │  Trained Model   │
                          │  (Checkpoint)    │
                          └────────┬─────────┘
                                   │
                                   ▼
                          ┌──────────────────┐
                          │  Inference       │
                          │  (Predictions)   │
                          └──────────────────┘
```

---

## Key Components Explained

### 1. **Transformer Encoder**
- **Purpose**: Processes the input sequence and learns temporal relationships
- **Mechanism**: Self-attention allows each frame to attend to all other frames
- **Output**: Rich encoded representations of the past 10 frames

### 2. **Transformer Decoder**
- **Purpose**: Predicts future frames based on encoded past frames
- **Mechanism**: Cross-attention (queries attend to memory) + Self-attention
- **Output**: Predictions for 20 future frames

### 3. **Multi-Head Attention**
- **8 attention heads**: Each head learns different types of relationships
- **Parallel processing**: All frames processed simultaneously (not sequential)
- **Long-range dependencies**: Can relate Frame 0 directly to Frame 9

### 4. **Feature Concatenation**
- **Hand motion**: 126 features (both hands)
- **Object context**: 24 features (box vertices)
- **Total**: 150 features per frame
- **Model sees both together**: Learns hand-object relationships

---

## Why This Architecture?

1. **Self-Attention**: Captures relationships between any two frames (not just adjacent)
2. **Parallel Processing**: Faster than sequential models (RNNs/LSTMs)
3. **Multi-Modal**: Processes hand and object features together
4. **Temporal Modeling**: Learns complex patterns across time
5. **Future Prediction**: Decoder generates multiple future frames at once

---

## Model Specifications

```
Architecture Type: Transformer Encoder-Decoder
Input Dimension: 150 features (hands + box)
Output Dimension: 63 features (receiving hand only)
Model Dimension: 512
Attention Heads: 8
Layers: 4 (encoder) + 4 (decoder)
Sequence Length: 10 frames input
Future Frames: 20 frames output
Parameters: ~2-3 million (estimated)
```

