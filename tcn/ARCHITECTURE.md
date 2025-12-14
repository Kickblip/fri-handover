# TCN (Temporal Convolutional Network) Architecture

This document describes the architecture of the Temporal Convolutional Network (TCN) implemented for predicting human-to-human handover trajectories. The model uses a series of dilated 1D convolutions to process a fixed window of past hand/object coordinates and predict the future trajectory of the receiving hand.

## 1. High-Level Overview

The TCN is designed as a **sequence-to-sequence** predictor. It takes a history of motion data and outputs a future horizon of predictions in a single forward pass ("Direct Prediction").

* **Input:** A sequence of `SEQ_LEN` frames (10 frames).
    * Features per frame: 150 (Giving Hand + Receiving Hand + Object).
* **Output:** A sequence of `FUTURE_FRAMES` (20 frames) for the Receiving Hand only.
    * Features per frame: 63 (21 landmarks × 3 coordinates).
* **Objective:** Learn the mapping $f(X_{t-10:t}) \rightarrow Y_{t+1:t+20}$ to forecast where the receiver's hand will move.

---

## 2. Model Architecture (`model.py`)

The core model is defined in `TemporalConvNet`. It consists of a stack of residual blocks followed by a linear decoding head.



### 2.1. The Building Block: `TemporalBlock`
The fundamental unit of the network is the `TemporalBlock`. It processes the input sequence while maintaining the temporal resolution.

* **Structure:** Two dilated convolution layers with a residual connection.
    1.  **Dilated Conv1d:** Expands the receptive field exponentially.
    2.  **Weight Normalization:** Applied to weights for training stability.
    3.  **Chomp1d:** Removes padding from the *end* of the sequence to ensure **causality** (no leaking information from the future).
    4.  **ReLU:** Activation function.
    5.  **Dropout:** Regularization (`0.3` probability).
    6.  **Residual Connection:** `Output = Conv2(Conv1(x)) + x`.

### 2.2. The Backbone: `TemporalConvNet`
The backbone stacks multiple `TemporalBlock` layers.
* **Input Layer:** Projects input dimension (150) to hidden dimension (1024).
* **Hidden Layers:** 4 stacked `TemporalBlock`s.
    * **Hidden Channels:** `[1024, 1024, 1024, 1024]`.
    * **Kernel Size:** `3`.
    * **Dilation:** Increases exponentially ($2^0, 2^1, 2^2, 2^3$).
    * **Receptive Field:** Covers the 20-frame input window.

### 2.3. The Decoding Head
This TCN uses a "Direct Prediction" head to generate all future frames at once.

1.  **Context Extraction:** The network processes the sequence and produces an output of shape `[Batch, Channels, Length]`. We slice the **last timestep** (`t=-1`) to get a single context vector.
2.  **Linear Projection:** A Linear layer projects this context vector to `FUTURE_FRAMES * Output_Dim` ($20 \times 63 = 1260$).
3.  **Reshape:** The flat vector is reshaped into `[Batch, 20, 63]` for the final prediction.

---

## 3. Configuration (`tcn_config.py`)

The model is configured with the "Champion" settings found during experimentation:

* **`TCN_HIDDEN_CHANNELS`**: `[1024, 1024, 1024, 1024]` (Wide network for capacity).
* **`TCN_KERNEL_SIZE`**: `3` (Local context focus).
* **`TCN_DROPOUT`**: `0.3` (Strong regularization).
* **`SEQ_STRIDE`**: `1` (Maximum data augmentation).
* **`BATCH_SIZE`**: `32` (Small batch for better generalization).

---

## 4. Data Pipeline (`data.py`)

The input data undergoes rigorous cleaning before training.

* **Consistency Check:** The `_ensure_hand_consistency` function detects if the tracking system swapped "Hand 0" and "Hand 1" (a common error) by tracking centroid movement. It swaps them back to ensure consistency.
* **Roles:**
    * **Giver (Input):** Hand 0 (Indices 0-63).
    * **Receiver (Target):** Hand 1 (Indices 63-126).
    * **Object:** Box corners (Indices 126-150).

---

## 5. Training & Inference

* **Training (`train.py`):**
    * **Loss:** MSE Loss.
    * **Optimizer:** AdamW with `LR=2e-4`.
    * **Scheduler:** `ReduceLROnPlateau` reduces LR if validation loss stalls.
    * **Early Stopping:** Stops training if validation loss doesn't improve for 5 epochs.

* **Inference (`infer.py`):**
    * **Strategy:** "Block Prediction". Instead of sliding frame-by-frame (which causes jitter), the model predicts a full 20-frame block, holds it, and then jumps forward. This produces smooth, stable trajectories.
    * **Visualization:** Uses Azure Kinect intrinsics (from `.mkv` metadata) to project 3D predictions onto the 2D video for accurate verification.
    * **Outputs:** Generates both a video overlay (`.mp4`) and a raw coordinate file (`_predictions.csv`).