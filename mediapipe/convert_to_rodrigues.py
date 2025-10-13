from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation
from typing import Literal, NamedTuple, Optional, Tuple, List

# ==== USER CONFIG ====
input_csv = Path("dataset/mediapipe_outputs/csv/2_w_b.csv")
output_csv = Path("dataset/mediapipe_outputs/csv/2_w_b_rodrigues.csv")
MAX_HANDS = 2
# =====================

class HandAxes(NamedTuple):
    ORIGIN: Literal['WRIST'] = 'WRIST'
    FORWARD_TARGET: Literal['MIDDLE_FINGER_MCP'] = 'MIDDLE_FINGER_MCP'
    UP_HINT_TARGET: Literal['INDEX_FINGER_MCP'] = 'INDEX_FINGER_MCP'

AXES = HandAxes()

def lm_cols(base: str, idx: int) -> Tuple[str, str, str]:
    b = base.lower()
    return (f"{b}_world_x_{idx}", f"{b}_world_y_{idx}", f"{b}_world_z_{idx}")

def get_lm(row: pd.Series, base: str, idx: int) -> Optional[np.ndarray]:
    try:
        x_col, y_col, z_col = lm_cols(base, idx)
        x, y, z = row[x_col], row[y_col], row[z_col]
        if np.isnan(x) or np.isnan(y) or np.isnan(z):
            return None
        return np.array([x, y, z], dtype=float)
    except KeyError:
        return None

def compute_rotvec(row: pd.Series, idx: int) -> Optional[np.ndarray]:
    # Skip if no label for this hand
    if pd.isna(row.get(f"hand_label_{idx}", np.nan)):
        return None

    O  = get_lm(row, AXES.ORIGIN, idx)
    Ft = get_lm(row, AXES.FORWARD_TARGET, idx)
    Ut = get_lm(row, AXES.UP_HINT_TARGET, idx)
    if O is None or Ft is None or Ut is None:
        return None

    F = Ft - O
    U = Ut - O
    nF, nU = np.linalg.norm(F), np.linalg.norm(U)
    if nF == 0 or nU == 0:
        return None

    # Build local frame
    X = F / nF
    Z = np.cross(X, U)
    nZ = np.linalg.norm(Z)
    if nZ == 0:
        return None
    Z = Z / nZ
    Y = np.cross(Z, X)
    nY = np.linalg.norm(Y)
    if nY == 0:
        return None
    Y = Y / nY

    # Local->World rotation
    R = np.column_stack((X, Y, Z))
    try:
        rot = Rotation.from_matrix(R)
    except ValueError:
        return None

    # Rodrigues rotation vector (x, y, z), |v| = angle in radians
    return rot.as_rotvec()

def main() -> None:
    if not input_csv.exists():
        raise FileNotFoundError(input_csv)

    df = pd.read_csv(input_csv)

    # Ensure required columns
    req = ["time_sec", "frame_index"]
    for i in range(MAX_HANDS):
        req += [f"hand_label_{i}", f"hand_score_{i}"]
    for c in req:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    out_rows: List[dict] = []
    for _, row in df.iterrows():
        rec = {
            "time_sec": row["time_sec"],
            "frame_index": int(row["frame_index"]),
            "hand_label_0": row.get("hand_label_0", np.nan),
            "hand_label_1": row.get("hand_label_1", np.nan),
            "hand_score_0": row.get("hand_score_0", np.nan),
            "hand_score_1": row.get("hand_score_1", np.nan),
            "rot_vec_x_0": np.nan, "rot_vec_y_0": np.nan, "rot_vec_z_0": np.nan,
            "rot_vec_x_1": np.nan, "rot_vec_y_1": np.nan, "rot_vec_z_1": np.nan,
        }

        for i in range(MAX_HANDS):
            v = compute_rotvec(row, i)
            if v is not None:
                rec[f"rot_vec_x_{i}"] = float(v[0])
                rec[f"rot_vec_y_{i}"] = float(v[1])
                rec[f"rot_vec_z_{i}"] = float(v[2])

        out_rows.append(rec)

    out_df = pd.DataFrame(out_rows, columns=[
        "time_sec","frame_index",
        "hand_label_0","hand_label_1","hand_score_0","hand_score_1",
        "rot_vec_x_0","rot_vec_y_0","rot_vec_z_0",
        "rot_vec_x_1","rot_vec_y_1","rot_vec_z_1"
    ])
    out_df.to_csv(output_csv, index=False)
    print(f"✅ Wrote wide Rodrigues vectors to: {output_csv.resolve()}")
    print(f"Rows: {len(out_df)}")

if __name__ == "__main__":
    main()