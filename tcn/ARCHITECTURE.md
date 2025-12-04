# TCN (Temporal Convolutional Network) Architecture

This document describes the architecture of the Temporal Convolutional Network (TCN) implemented for predicting human-to-human handover trajectories. The model uses a series of dilated 1D convolutions to process a sequence of past hand/object coordinates and predict the future trajectory of the receiving hand.

## 1. High-Level Overview

The TCN is designed as a **sequence-to-sequence** predictor. It takes a fixed history window of motion data and outputs a fixed future horizon of predictions in a single forward pass (Direct Prediction).

* **Input:** A sequence of `SEQ_LEN` frames (default: 20).
    * Features per frame: 150 (Giving Hand + Receiving Hand + Object).
* **Output:** A sequence of `FUTURE_FRAMES` (default: 10) for the Receiving Hand only.
    * Features per frame: 63 (21 landmarks × 3 coordinates).
* **Prediction Strategy:** Direct Multi-Horizon. The model effectively learns a function $f(X_{t-20:t}) \rightarrow Y_{t+1:t+10}$.

---

## 2. Model Architecture (`model.py`)

The core model is defined in `TemporalConvNet`. It consists of a stack of residual blocks followed by a linear decoding head.



### 2.1. The Building Block: `TemporalBlock`
The fundamental unit of the network is the `TemporalBlock`. Each block processes the input sequence while preserving its length but increasing its receptive field.

* **Structure:** Two dilated convolution layers with a residual connection.
    1.  **Dilated Conv1d:** Expands the receptive field without losing resolution.
    2.  **Weight Normalization:** Applied to all convolution weights for training stability.
    3.  **Chomp1d:** Removes padding from the *end* of the sequence to ensures **causality** (no leaking information from the future).
    4.  **ReLU:** Activation function.
    5.  **Dropout:** Regularization.
    6.  **Residual Connection:** The input `x` is added to the output of the two conv layers: `Output = Conv2(Conv1(x)) + x`. If channel dimensions change, a 1x1 convolution aligns them.

### 2.2. The Backbone: `TemporalConvNet`
The backbone stacks multiple `TemporalBlock` layers. The dilation factor increases exponentially with depth ($d = 2^i$) to capture long-term dependencies.

* **Input Layer:** Projects input dimension (150) to hidden dimension (e.g., 1024).
* **Hidden Layers:** 4 stacked `TemporalBlock`s.
    * **Dilation Schedule:** 1, 2, 4, 8.
    * **Receptive Field:** The model can "see" effective history up to $(K-1) \times \sum d_i + 1$. With Kernel=3 and 4 layers, this covers the entire 20-frame history comfortably.
* **Layer Normalization:** Not explicitly used inside blocks, relies on Weight Norm (0.01).

### 2.3. The Decoding Head
Unlike autoregressive models that loop, this TCN uses a "Direct Prediction" head to generate all future frames at once.

1.  **Context Extraction:** The network processes the entire sequence $L$ and produces output of shape `[Batch, Channels, Length]`. We slice the **last timestep** (`t=-1`) to get a single context vector that summarizes the entire history window.
2.  **Linear Projection:** A simple Linear layer projects this context vector from `Hidden_Channels` (e.g., 256) to `FUTURE_FRAMES * Output_Dim` (e.g., $10 \times 63 = 630$).
3.  **Reshape:** The flat vector is reshaped into `[Batch, 10, 63]` to form the final trajectory prediction.

---

## 3. Configuration (`tcn_config.py`)

The model's shape and behavior are controlled by `tcn_config.py`:

* **`TCN_HIDDEN_CHANNELS`:** `[1024, 1024, 1024, 1024]` (4 layers of width 256).
* **`TCN_KERNEL_SIZE`:** `3` (Each conv looks at 3 timesteps).
* **`TCN_DROPOUT`:** `0.3` (Probability of zeroing elements).
* **`SEQ_LEN`:** `20` (Input window size).
* **`FUTURE_FRAMES`:** `10` (Prediction horizon).

---

## 4. Data Pipeline (`data.py`)

The input data handles complex tracking scenarios before reaching the model.

* **Inputs:** * `h0` (Giving Hand): 63 features.
    * `h1` (Receiving Hand): 63 features.
    * `box` (Object): 24 features (8 vertices × 3 coords).
* **Consistency Check:** The `_ensure_hand_consistency` function prevents "hand switching" errors by tracking landmark centroids frame-to-frame. If `h0` jumps to `h1`'s position, they are swapped back to ensure the model trains on a consistent identity.
* **Target Selection:** The receiving hand (typically indices 63-126) is extracted as the training target.

---

## 5. Training & Inference

* **Training (`train.py`):** Optimizes the model using MSE Loss and the AdamW optimizer. It uses a `ReduceLROnPlateau` scheduler to lower the learning rate if validation loss stalls.
* **Inference (`infer.py`):**
    * **Block Prediction Strategy:** Instead of a sliding window (predicting every single frame), the inference loop jumps forward by `FUTURE_FRAMES`.
    * It predicts a 20-frame block, locks it in, and then predicts the next block. This results in smoother visualizations compared to jittery frame-by-frame updates.
    * **Visualization:** Uses Azure Kinect intrinsics (if available) or estimations to project 3D predictions back onto the 2D video for evaluation.