# Model Inputs - Detailed Explanation

## Overview

The model takes sequences of **multi-modal features** as input, combining hand tracking data and object tracking data.

---

## Input Shape

```
Input: [Batch, Sequence_Length, Feature_Dimension]

Where:
- Batch: Number of samples (typically 64 during training)
- Sequence_Length: 10 frames (SEQ_LEN = 10, ~0.33 seconds @ 30 FPS)
- Feature_Dimension: Variable (typically 126-150 features per frame)
```

---

## Feature Components

The input features are **concatenated** from three sources:

### 1. Hand Coordinates (126 features) - **REQUIRED**

From `load_both_hands_world()`:

```
Hand 0 (Receiving Hand): 63 features
├─ Landmark 0 (Wrist):     x, y, z
├─ Landmark 1 (Thumb CMC): x, y, z
├─ Landmark 2 (Thumb MCP): x, y, z
├─ Landmark 3 (Thumb IP):  x, y, z
├─ Landmark 4 (Thumb Tip): x, y, z
├─ Landmark 5 (Index MCP): x, y, z
├─ Landmark 6 (Index PIP): x, y, z
├─ Landmark 7 (Index DIP): x, y, z
├─ Landmark 8 (Index Tip): x, y, z
├─ Landmark 9 (Middle MCP): x, y, z
├─ Landmark 10 (Middle PIP): x, y, z
├─ Landmark 11 (Middle DIP): x, y, z
├─ Landmark 12 (Middle Tip): x, y, z
├─ Landmark 13 (Ring MCP): x, y, z
├─ Landmark 14 (Ring PIP): x, y, z
├─ Landmark 15 (Ring DIP): x, y, z
├─ Landmark 16 (Ring Tip): x, y, z
├─ Landmark 17 (Pinky MCP): x, y, z
├─ Landmark 18 (Pinky PIP): x, y, z
├─ Landmark 19 (Pinky DIP): x, y, z
└─ Landmark 20 (Pinky Tip): x, y, z

Hand 1 (Giving Hand): 63 features
└─ Same structure as Hand 0 (21 landmarks × 3 coordinates)

Total Hand Features: 63 + 63 = 126 features
```

**Data Source**: CSV files with columns like:
- `h0_lm0_x`, `h0_lm0_y`, `h0_lm0_z` (hand 0, landmark 0)
- `h0_lm1_x`, `h0_lm1_y`, `h0_lm1_z` (hand 0, landmark 1)
- ... (continues for all 21 landmarks of hand 0)
- `h1_lm0_x`, `h1_lm0_y`, `h1_lm0_z` (hand 1, landmark 0)
- ... (continues for all 21 landmarks of hand 1)

**Coordinate System**: World coordinates (x, y, z) in meters, obtained by:
1. MediaPipe detects 2D hand landmarks in image space
2. Azure Kinect provides depth at each landmark
3. Calibration converts (pixel, depth) → (x, y, z) world coordinates

---

### 2. Box Coordinates (0-24 features) - **OPTIONAL**

From `load_box_coordinates()`:

```
Box Vertices: Variable number of features
├─ Typically 8 vertices × 3 coordinates = 24 features
├─ Vertex 0: v0_x, v0_y, v0_z
├─ Vertex 1: v1_x, v1_y, v1_z
├─ Vertex 2: v2_x, v2_y, v2_z
├─ Vertex 3: v3_x, v3_y, v3_z
├─ Vertex 4: v4_x, v4_y, v4_z
├─ Vertex 5: v5_x, v5_y, v5_z
├─ Vertex 6: v6_x, v6_y, v6_z
└─ Vertex 7: v7_x, v7_y, v7_z

If box file doesn't exist: 0 features (empty array)
```

**Data Source**: CSV files with columns like:
- `v0_x`, `v0_y`, `v0_z` (vertex 0 coordinates)
- `v1_x`, `v1_y`, `v1_z` (vertex 1 coordinates)
- ... (continues for all vertices)

**How it's obtained**:
1. AprilTag detection finds tags on the object
2. Tag pose estimation gives object position/orientation
3. Box vertices computed from tag pose + known box dimensions
4. Vertices converted to world coordinates

**Note**: If no box file exists, this component is 0 features (the model still works with just hand data).

---

### 3. Vertices (0-N features) - **OPTIONAL**

From `load_vertices()`:

```
Additional vertex features: Variable number
└─ Any numeric columns from {stem}_vertices.csv
   (Legacy feature, typically 0 features)
```

**Data Source**: CSV files in `VERTICES_DIR` (legacy format)

**Note**: Typically 0 features in current setup.

---

## Feature Concatenation

From `load_features()` in `model/data.py`:

```python
def load_features(stem: str) -> Tuple[np.ndarray, List[int]]:
    """
    Load input features: both hands world coordinates + box coordinates + optional vertices.
    Returns concatenated features: [hands || box || vertices]
    """
    X_h, frames = load_both_hands_world(stem)  # [T, 126]
    X_b = load_box_coordinates(stem, frames)  # [T, D_box] where D_box can be 0-24
    X_v = load_vertices(stem, frames)        # [T, D_vertices] typically 0
    X   = np.concatenate([X_h, X_b, X_v], axis=1)  # [T, 126 + D_box + D_vertices]
    return X, frames
```

**Concatenation Order**:
```
[Hand 0 (63) || Hand 1 (63) || Box (0-24) || Vertices (0-N)]
```

---

## Complete Input Example

### For One Frame:

```
Frame t features = [
  # Hand 0 (Receiving Hand) - 63 features
  h0_lm0_x, h0_lm0_y, h0_lm0_z,    # Wrist
  h0_lm1_x, h0_lm1_y, h0_lm1_z,    # Thumb CMC
  h0_lm2_x, h0_lm2_y, h0_lm2_z,    # Thumb MCP
  ... (21 landmarks total)
  
  # Hand 1 (Giving Hand) - 63 features
  h1_lm0_x, h1_lm0_y, h1_lm0_z,    # Wrist
  h1_lm1_x, h1_lm1_y, h1_lm1_z,    # Thumb CMC
  ... (21 landmarks total)
  
  # Box Vertices - 24 features (if available)
  v0_x, v0_y, v0_z,                 # Box vertex 0
  v1_x, v1_y, v1_z,                 # Box vertex 1
  ... (8 vertices total)
  
  # Additional vertices - 0 features (typically)
]

Total: 126 (hands) + 24 (box) = 150 features per frame
```

### For One Sequence (Training Input):

```
Input Sequence: [10 frames × 150 features]

Frame 0: [150 features]
Frame 1: [150 features]
Frame 2: [150 features]
...
Frame 9: [150 features]

Shape: [10, 150]
```

### For One Batch (Training):

```
Batch Input: [64 sequences × 10 frames × 150 features]

Shape: [64, 10, 150]
```

---

## Input Dimensions Summary

| Component | Features | Required? | Description |
|-----------|----------|-----------|-------------|
| Hand 0 (Receiving) | 63 | ✅ Yes | 21 landmarks × 3 coords (x, y, z) |
| Hand 1 (Giving) | 63 | ✅ Yes | 21 landmarks × 3 coords (x, y, z) |
| Box Vertices | 0-24 | ❌ Optional | 8 vertices × 3 coords (if available) |
| Additional Vertices | 0-N | ❌ Optional | Legacy feature (typically 0) |
| **Total** | **126-150** | - | Depends on box/vertex availability |

**Most Common**: 126 features (hands only) or 150 features (hands + box)

---

## Coordinate System

All coordinates are in **world space** (meters):
- **Origin**: Camera coordinate system origin
- **Units**: Meters
- **Axes**: 
  - X: Typically right (from camera perspective)
  - Y: Typically down (from camera perspective)
  - Z: Typically forward (depth, away from camera)

**Conversion Process**:
1. MediaPipe detects 2D landmarks in image space (pixels)
2. Azure Kinect provides depth at each landmark (millimeters)
3. Camera calibration converts: `(u, v, depth_mm) → (x, y, z) meters`

---

## Sequence Structure

The model processes **sequences** of frames, not individual frames:

```
Sequence Length: 10 frames (SEQ_LEN = 10)
Time Span: ~0.33 seconds at 30 FPS

Input Sequence:
┌─────────────────────────────────────────┐
│ Frame 0: [150 features]                │ ← t=0
│ Frame 1: [150 features]                │ ← t=1
│ Frame 2: [150 features]                │ ← t=2
│ ...                                    │
│ Frame 9: [150 features]                │ ← t=9
└─────────────────────────────────────────┘
         ↓
    Transformer Model
         ↓
┌─────────────────────────────────────────┐
│ Predicted Frame 10: [63 features]      │ ← t=10
│ Predicted Frame 11: [63 features]      │ ← t=11
│ ...                                    │
│ Predicted Frame 29: [63 features]      │ ← t=29
└─────────────────────────────────────────┘
```

---

## Code Reference

**Key Functions**:
- `load_features()` - Main function that loads and concatenates all features
- `load_both_hands_world()` - Loads hand coordinates (126 features)
- `load_box_coordinates()` - Loads box coordinates (0-24 features)
- `load_vertices()` - Loads additional vertices (0-N features)

**Location**: `model/data.py`

**Configuration**: `model/config.py`
- `SEQ_LEN = 10` (input sequence length)
- `FUTURE_FRAMES = 20` (output sequence length)

---

## Important Notes

1. **Hand Order**: 
   - Hand 0 = Receiving hand (what we predict)
   - Hand 1 = Giving hand (context for prediction)

2. **Missing Data**: 
   - Missing hand landmarks → filled with 0.0
   - Missing box data → 0 features (model still works)
   - NaN values → converted to 0.0

3. **Hand Consistency**: 
   - The system includes a hand consistency algorithm to prevent hand identity switching
   - Ensures Hand 0 stays Hand 0 throughout the video

4. **Feature Ordering**:
   - Hand landmarks are sorted alphabetically (ensures consistent ordering)
   - Box vertices are sorted alphabetically
   - Concatenation order: [Hand 0 || Hand 1 || Box || Vertices]

5. **Dynamic Input Dimension**:
   - The model adapts to the actual input dimension
   - If box data is missing, input_dim = 126
   - If box data exists, input_dim = 150 (typically)

