from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation
from typing import List

# ==== USER CONFIG ====
# Input: wide quaternions CSV you created earlier
input_quat_csv = Path("dataset/mediapipe_outputs/csv/2_w_b_quaternions.csv")
# Output: wide Rodrigues CSV
output_rodrigues_csv = Path("dataset/mediapipe_outputs/csv/2_w_b_rodrigues_from_quat.csv")
MAX_HANDS = 2
# =====================

def quat_to_rotvec(w: float, x: float, y: float, z: float) -> List[float]:
    """Convert a single quaternion (w, x, y, z) to Rodrigues vector [rx, ry, rz]."""
    if any(map(lambda v: pd.isna(v), [w, x, y, z])):
        return [np.nan, np.nan, np.nan]
    # SciPy expects [x, y, z, w]
    r = Rotation.from_quat([x, y, z, w]).as_rotvec()
    return r.tolist()

def main() -> None:
    if not input_quat_csv.exists():
        raise FileNotFoundError(f"Input quaternion CSV not found: {input_quat_csv}")

    df = pd.read_csv(input_quat_csv)

    # Ensure expected columns are present
    required = ["time_sec", "frame_index",
                "hand_label_0", "hand_label_1", "hand_score_0", "hand_score_1"]
    for i in range(MAX_HANDS):
        required += [f"quat_w_{i}", f"quat_x_{i}", f"quat_y_{i}", f"quat_z_{i}"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input: {missing}")

    out_rows = []
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
            w = row[f"quat_w_{i}"]; x = row[f"quat_x_{i}"]; y = row[f"quat_y_{i}"]; z = row[f"quat_z_{i}"]
            if pd.isna(w):
                # leave NaNs if quaternion not present for this hand
                continue
            rx, ry, rz = quat_to_rotvec(w, x, y, z)
            rec[f"rot_vec_x_{i}"] = rx
            rec[f"rot_vec_y_{i}"] = ry
            rec[f"rot_vec_z_{i}"] = rz

        out_rows.append(rec)

    out_df = pd.DataFrame(out_rows, columns=[
        "time_sec","frame_index",
        "hand_label_0","hand_label_1","hand_score_0","hand_score_1",
        "rot_vec_x_0","rot_vec_y_0","rot_vec_z_0",
        "rot_vec_x_1","rot_vec_y_1","rot_vec_z_1"
    ])
    out_df.to_csv(output_rodrigues_csv, index=False)
    print(f"✅ Wrote wide Rodrigues vectors to: {output_rodrigues_csv.resolve()}")
    print(f"Rows: {len(out_df)}")

if __name__ == "__main__":
    main()