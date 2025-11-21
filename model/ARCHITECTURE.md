# Model Folder Architecture

This document explains how the `model/` folder works and how each file interacts with each other.

## 📁 File Structure

```
model/
├── __init__.py          # Package initialization
├── config.py           # Central configuration (paths, hyperparameters)
├── data.py             # Data loading and preprocessing
├── model.py            # Neural network architecture
├── train.py            # Training script
├── infer.py            # Inference and video generation
├── eval.py             # Evaluation metrics (legacy - for classification)
├── utils.py            # Utility functions (seeding, metrics)
└── viz_open3d.py       # 3D visualization (Open3D)
```

## 🔄 Data Flow

### 1. **Configuration (`config.py`)**
   - **Purpose**: Central hub for all paths and hyperparameters
   - **Exports**: 
     - Paths: `WORLD_DIR`, `VERTICES_DIR`, `CKPT_DIR`, `PRED_DIR`, `VIDEO_DIR`
     - Hyperparameters: `SEQ_LEN`, `BATCH_SIZE`, `FUTURE_FRAMES`, `D_MODEL`, etc.
   - **Used by**: All other files import from here

### 2. **Data Loading (`data.py`)**
   - **Purpose**: Load and preprocess data from CSV files
   - **Key Functions**:
     - `load_both_hands_world()`: Load world coordinates (x, y, z) for both hands (concatenated)
     - `load_vertices()`: Load object vertex features (optional)
     - `load_features()`: Combine both hands' world coordinates + optional vertices → input features
     - `load_receiving_hand_world()`: Load receiving hand (hand_1) world coordinates → targets
     - `make_sequences_with_targets()`: Convert per-frame data into sliding windows
     - `HandoverDataset`: PyTorch Dataset class
     - `build_loaders()`: Create train/val/test DataLoaders
   
   - **Data Flow**:
     ```
     CSV files → load_*() functions → numpy arrays → sequences → Dataset → DataLoader
     ```

### 3. **Model Architecture (`model.py`)**
   - **Purpose**: Define the Transformer encoder-decoder network
   - **Key Components**:
     - `PositionalEncoding`: Adds positional information to sequences
     - `HandoverTransformer`: Main model
       - Encoder: Processes input sequence (both hands + vertices)
       - Decoder: Predicts future frames of receiving hand
   
   - **Architecture**:
     ```
     Input [B, L, D_in] 
       → Project to d_model
       → Add positional encoding
       → Encoder (self-attention)
       → Decoder (with learnable queries for future frames)
       → Output [B, future_frames, D_out]
     ```

### 4. **Training (`train.py`)**
   - **Purpose**: Train the model on data
   - **Flow**:
     1. Load data using `build_loaders()` from `data.py`
     2. Create model using `HandoverTransformer` from `model.py`
     3. Train loop: forward pass → MSE loss → backprop → update weights
     4. Validation: evaluate on val set, save best model
     5. Save checkpoint to `CKPT_PATH` (from `config.py`)
   
   - **Checkpoint Format**:
     ```python
     {
         "model": state_dict,
         "in_dim": int,
         "out_dim": int,
         "cfg": {D_MODEL, N_HEAD, ...}
     }
     ```

### 5. **Inference (`infer.py`)**
   - **Purpose**: Make predictions and generate visualization videos
   - **Key Functions**:
     - `load_checkpoint()`: Load trained model from checkpoint
     - `predict_future_frames()`: Slide window over sequence, predict future frames
     - `load_giving_hand_world()`: Load giving hand (hand_0) coordinates
     - `create_video()`: Overlay predictions on original video
   
   - **Flow**:
     ```
     Load checkpoint → Load input features → Predict → Save CSV → Generate video
     ```

## 🔗 File Interactions

### Training Pipeline
```
train.py
  ├─→ config.py (imports paths, hyperparameters)
  ├─→ data.py (build_loaders, load_features, load_receiving_hand_world)
  └─→ model.py (HandoverTransformer)
       └─→ Saves checkpoint to config.CKPT_PATH
```

### Inference Pipeline
```
infer.py
  ├─→ config.py (imports paths, hyperparameters)
  ├─→ data.py (load_features, load_receiving_hand_world, _read_csv, _pick_col)
  ├─→ model.py (HandoverTransformer)
  └─→ Loads checkpoint from config.CKPT_PATH
```

### Data Flow Diagram
```
CSV Files (dataset/)
  ↓
data.py (load_both_hands_world, load_vertices, load_receiving_hand_world)
  ↓
HandoverDataset (sequences with targets)
  ↓
DataLoader (batches)
  ↓
train.py → model.py (HandoverTransformer)
  ↓
Checkpoint (model_output/checkpoints/)
  ↓
infer.py → predict_future_frames()
  ↓
Predictions CSV + Video (model_output/)
```

## 📊 Key Data Structures

### Input Features
- **Shape**: `[T, D_in]` where T = number of frames
- **Content**: 
  - Hand 0 world coordinates (x, y, z for 21 landmarks = 63 features)
  - Hand 1 world coordinates (x, y, z for 21 landmarks = 63 features)
  - Object vertices (optional, may be empty)
  - Total: 126 features (both hands) + optional vertices

### Target Features
- **Shape**: `[T, D_out]` where D_out = 63 (21 landmarks × 3 coords)
- **Content**: Receiving hand (hand_1) world coordinates

### Sequences
- **Input**: `[N, SEQ_LEN, D_in]` - sliding windows of input
- **Target**: `[N, FUTURE_FRAMES, D_out]` - future frames to predict

### Model Output
- **Shape**: `[B, FUTURE_FRAMES, D_out]`
- **Content**: Predicted receiving hand world coordinates for future frames

## 🎯 Main Workflows

### 1. Training Workflow
```bash
python -m model.train
```
1. `train.py` imports config, data, model
2. Builds data loaders from CSV files
3. Creates model with config hyperparameters
4. Trains for MAX_EPOCHS epochs
5. Saves best model to checkpoint

### 2. Inference Workflow
```bash
python -m model.infer <stem> --video
```
1. `infer.py` loads checkpoint
2. Loads input features for stem
3. Slides window, makes predictions
4. Saves predictions to CSV
5. (Optional) Generates video with predictions overlaid

## 🔧 Configuration Points

All configuration is centralized in `config.py`:
- **Data paths**: Where to find input CSVs
- **Model size**: D_MODEL, N_HEAD, N_LAYERS, etc.
- **Training**: LR, MAX_EPOCHS, BATCH_SIZE
- **Sequence**: SEQ_LEN, SEQ_STRIDE, FUTURE_FRAMES

## 📝 Notes

- **Stem**: Video identifier (e.g., "1_w_b") used to find corresponding CSV files
- **Hand 0**: Giving hand (the one giving the object)
- **Hand 1**: Receiving hand (the one receiving the object) - this is what we predict
- **Future Frames**: Number of frames ahead to predict (default: 5)
- **Sequence Length**: Input window size (default: 30 frames ≈ 1 second @ 30fps)

