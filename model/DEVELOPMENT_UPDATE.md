# Transformer Development Update

## 📅 Current Status

**Date**: [Current Date]  
**Status**: ✅ Fully Functional  
**Model**: Handover Transformer for Future Frame Prediction

---

## 🎯 Project Overview

The Handover Transformer predicts future frames of the receiving hand during handover events using a Transformer encoder-decoder architecture. The model takes sequences of both hands' features and predicts the receiving hand's world coordinates for the next 5 frames.

---

## ✅ Completed Features

### 1. **Core Architecture** (`model.py`)
- ✅ Transformer encoder-decoder architecture
- ✅ Positional encoding for input sequences
- ✅ Learnable query embeddings for future frame prediction
- ✅ Configurable model size (d_model=256, n_layers=4, n_heads=8)
- ✅ GELU activation and layer normalization

### 2. **Data Pipeline** (`data.py`)
- ✅ Support for wide-format CSV files (hand_0 and hand_1 features)
- ✅ Automatic detection of numeric vs non-numeric columns
- ✅ Loading of world coordinates (x, y, z Cartesian coordinates) for both hands
- ✅ Optional vertex features support
- ✅ Sliding window sequence generation
- ✅ Train/validation/test splits (stem-wise to prevent leakage)
- ✅ Robust error handling for missing data

### 3. **Training System** (`train.py`)
- ✅ Full training loop with MSE loss
- ✅ Early stopping based on validation loss
- ✅ Learning rate scheduling (ReduceLROnPlateau)
- ✅ Gradient clipping for stability
- ✅ Checkpoint saving (best model)
- ✅ Support for training on specific stems

### 4. **Inference System** (`infer.py`)
- ✅ Model checkpoint loading
- ✅ Sliding window prediction over sequences
- ✅ CSV output for predictions
- ✅ **Video visualization with overlays** (NEW)
  - Predicted receiving hand (hand_1) 21 points in RED
  - Giving hand (hand_0) 21 points in GREEN
  - Overlaid on original video frames

### 5. **Configuration** (`config.py`)
- ✅ Centralized configuration for all paths and hyperparameters
- ✅ Organized output structure under `dataset/model_output/`
- ✅ Easy hyperparameter tuning

---

## 🏗️ Architecture Details

### Model Architecture
```
Input [B, L, D_in]
  ↓
Linear Projection → [B, L, d_model]
  ↓
Positional Encoding
  ↓
Transformer Encoder (4 layers, 8 heads)
  ↓
Transformer Decoder (with learnable queries for future frames)
  ↓
Layer Normalization
  ↓
Linear Projection → [B, future_frames, D_out]
```

### Key Hyperparameters
- **Sequence Length**: 30 frames (~1 second @ 30fps)
- **Future Frames**: 5 frames ahead
- **Model Dimension**: 256
- **Number of Layers**: 4 (encoder + decoder)
- **Number of Heads**: 8
- **FFN Dimension**: 512
- **Dropout**: 0.10
- **Learning Rate**: 2e-4
- **Batch Size**: 64

### Input/Output
- **Input**: Concatenated world coordinates (x, y, z) from both hands (126 features: 63 per hand) + optional vertices
- **Output**: World coordinates of receiving hand (hand_1) for 21 landmarks × 3 coords = 63 features per frame

---

## 📊 Recent Improvements

### Video Visualization Enhancement (Latest)
- **Feature**: Enhanced video output with dual-hand visualization
- **Implementation**: 
  - Loads original video from `dataset/input_video/`
  - Projects 3D world coordinates to 2D image coordinates
  - Overlays predicted receiving hand (RED) and giving hand (GREEN) on original video
- **Output**: Professional visualization showing both hands simultaneously

### Data Loading Robustness
- Improved handling of non-numeric columns
- Better error messages for missing data
- Support for multiple CSV formats (wide/long)

---

## 📁 Project Structure

```
model/
├── config.py          # Central configuration
├── data.py            # Data loading and preprocessing
├── model.py           # Transformer architecture
├── train.py           # Training script
├── infer.py           # Inference and video generation
├── utils.py           # Utility functions
├── eval.py            # Evaluation metrics (legacy)
└── ARCHITECTURE.md    # Detailed architecture documentation
```

---

## 🚀 Usage

### Training
```bash
python -m model.train
```

### Inference
```bash
# Generate predictions only
python -m model.infer 1_w_b

# Generate predictions + video visualization
python -m model.infer 1_w_b --video
```

### Outputs
- **Checkpoints**: `dataset/model_output/checkpoints/handover_transformer.pt`
- **Predictions**: `dataset/model_output/predictions/<stem>_future_predictions.csv`
- **Videos**: `dataset/model_output/videos/<stem>_predicted_future.mp4`

---

## 📈 Current Capabilities

1. ✅ **Training**: Full training pipeline with validation and early stopping
2. ✅ **Inference**: Predict future frames for any input sequence
3. ✅ **Visualization**: Generate videos with predicted and actual hand positions
4. ✅ **Data Handling**: Robust loading from CSV files with error handling
5. ✅ **Modularity**: Clean separation of concerns (config, data, model, training, inference)

---

## 🔄 Data Flow

```
CSV Files (World Coordinates)
  ↓
data.py (load and preprocess)
  ↓
HandoverDataset (create sequences)
  ↓
DataLoader (batches)
  ↓
train.py → model.py (HandoverTransformer)
  ↓
Checkpoint (saved model)
  ↓
infer.py (load checkpoint, predict)
  ↓
Predictions CSV + Visualization Video
```

---

## 🎓 Technical Highlights

1. **Encoder-Decoder Architecture**: Uses Transformer encoder to process input sequence, decoder to predict future frames
2. **Learnable Queries**: Future frame positions are learned through query embeddings
3. **Positional Encoding**: Both input sequences and future frames have positional information
4. **Stem-wise Splits**: Prevents data leakage by splitting at video level
5. **Robust Data Loading**: Handles missing data, non-numeric columns, and multiple formats

---

## 📝 Next Steps / Future Work

1. **Evaluation Metrics**: Implement quantitative evaluation metrics (MSE, MAE per landmark)
2. **Hyperparameter Tuning**: Experiment with different model sizes and architectures
3. **Data Augmentation**: Add augmentation techniques for better generalization
4. **Multi-step Prediction**: Extend to predict further into the future
5. **Real-time Inference**: Optimize for real-time prediction during handover

---

## 📧 Contact / Questions

For questions about the implementation, please refer to:
- `model/ARCHITECTURE.md` - Detailed architecture documentation
- `model/README.md` - Quick start guide
- Code comments in individual files

---

**Status**: ✅ Model is fully functional and ready for evaluation/testing

