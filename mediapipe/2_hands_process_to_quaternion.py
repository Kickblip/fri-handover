from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation
from typing import Literal, NamedTuple, Optional, Tuple, List

# ==== USER CONFIGURATION: QUATERNION SCRIPT (TWO-HANDS) ====
input_csv = Path("dataset/mediapipe_outputs/csv/2_w_b.csv")
output_quaternion_csv = Path("dataset/mediapipe_outputs/csv/2_hand_w_b_quaternions.csv")

MAX_HANDS = 2  # matches your extractor
# ===========================================================

class HandAxes(NamedTuple):
    ORIGIN: Literal['WRIST'] = 'WRIST'
    FORWARD_TARGET: Literal['MIDDLE_FINGER_MCP'] = 'MIDDLE_FINGER_MCP'
    UP_HINT_TARGET: Literal['INDEX_FINGER_MCP'] = 'INDEX_FINGER_MCP'

AXES_CONFIG = HandAxes()

def lm_cols(base: str, hand_idx: int) -> Tuple[str, str, str]:
    b = base.lower()
    return (f"{b}_world_x_{hand_idx}", f"{b}_world_y_{hand_idx}", f"{b}_world_z_{hand_idx}")

def get_landmark_coords(row: pd.Series, landmark: str, hand_idx: int) -> Optional[np.ndarray]:
    try:
        x_col, y_col, z_col = lm_cols(landmark, hand_idx)
        x, y, z = row[x_col], row[y_col], row[z_col]
        if np.isnan(x) or np.isnan(y) or np.isnan(z):
            return None
        return np.array([x, y, z], dtype=float)
    except KeyError:
        return None

def compute_quaternion_for_hand(row: pd.Series, hand_idx: int) -> Optional[np.ndarray]:
    """
    Compute (w, x, y, z) quaternion mapping the hand's local frame to world,
    built from:
      X (forward) = wrist -> middle_mcp
      Z (up)      = X x (wrist -> index_mcp)
      Y (right)   = Z x X
    """
    label_col = f"hand_label_{hand_idx}"
    score_col = f"hand_score_{hand_idx}"
    label = row.get(label_col, np.nan)
    score = row.get(score_col, np.nan)

    if pd.isna(label):
        return None

    # 1) Pull landmarks
    O = get_landmark_coords(row, AXES_CONFIG.ORIGIN, hand_idx)
    F_t = get_landmark_coords(row, AXES_CONFIG.FORWARD_TARGET, hand_idx)
    U_t = get_landmark_coords(row, AXES_CONFIG.UP_HINT_TARGET, hand_idx)
    if O is None or F_t is None or U_t is None:
        return None

    # 2) Vectors
    F_vec = F_t - O
    U_hint = U_t - O

    nF = np.linalg.norm(F_vec)
    nU = np.linalg.norm(U_hint)
    if nF == 0 or nU == 0:
        return None

    # 3) Orthonormal basis
    X = F_vec / nF
    Z = np.cross(X, U_hint)
    nZ = np.linalg.norm(Z)
    if nZ == 0:
        return None
    Z = Z / nZ
    Y = np.cross(Z, X)
    nY = np.linalg.norm(Y)
    if nY == 0:
        return None
    Y = Y / nY

    # Rotation matrix columns are the local axes in world coords
    R = np.column_stack((X, Y, Z))

    try:
        rot = Rotation.from_matrix(R)
    except ValueError:
        return None

    quat_xyzw = rot.as_quat()      # (x, y, z, w)
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
    return quat_wxyz

def main() -> None:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)

    required_meta = []
    for i in range(MAX_HANDS):
        required_meta += [f"hand_label_{i}", f"hand_score_{i}"]
    for col in ["time_sec", "frame_index"] + required_meta:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    records: List[dict] = []
    for _, row in df.iterrows():
        t = row["time_sec"]
        fidx = row["frame_index"]

        for i in range(MAX_HANDS):
            label = row.get(f"hand_label_{i}", np.nan)
            score = row.get(f"hand_score_{i}", np.nan)
            if pd.isna(label):
                continue

            quat = compute_quaternion_for_hand(row, i)
            if quat is None:
                continue

            w, x, y, z = quat.tolist()
            records.append({
                "time_sec": t,
                "frame_index": int(fidx),
                "hand_index": i,
                "hand_label": str(label).lower(),
                "hand_score": float(score) if not pd.isna(score) else np.nan,
                "quat_w": w,
                "quat_x": x,
                "quat_y": y,
                "quat_z": z,
            })

    if len(records) == 0:
        out = pd.DataFrame(columns=[
            "time_sec", "frame_index", "hand_index", "hand_label", "hand_score",
            "quat_w", "quat_x", "quat_y", "quat_z",
        ])
        out.to_csv(output_quaternion_csv, index=False)
        print("No valid quaternions computed; wrote header-only CSV.")
        return

    out = pd.DataFrame.from_records(records)
    out.sort_values(["frame_index", "hand_index"], inplace=True, kind="mergesort")
    out.to_csv(output_quaternion_csv, index=False)

    print(f"✅ Successfully computed quaternions for {len(out)} (frame, hand) detections.")
    print(f"➡️  {output_quaternion_csv.resolve()}")

if __name__ == "__main__":
    main()
