from pathlib import Path
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation
from typing import NamedTuple, Literal

# ==== USER CONFIGURATION: QUATERNION SCRIPT ====
input_csv = Path("outputs/hand_3D_world_landmarks.csv")  # Input from landmark extraction
output_quaternion_csv = Path("outputs/hand_quaternions.csv") # <-- Target output file
# ==========================================

# Define the landmarks to use for the local coordinate system
class HandAxes(NamedTuple):
    """Defines the hand landmarks used to construct the local coordinate system."""
    ORIGIN: Literal['WRIST'] = 'WRIST'
    # Vector from ORIGIN to FORWARD_TARGET defines the FORWARD direction (X-axis hint).
    FORWARD_TARGET: Literal['MIDDLE_FINGER_MCP'] = 'MIDDLE_FINGER_MCP'
    # Vector from ORIGIN to UP_HINT_TARGET is used in the cross-product to determine UP (Z-axis hint).
    UP_HINT_TARGET: Literal['INDEX_FINGER_MCP'] = 'INDEX_FINGER_MCP'

AXES_CONFIG = HandAxes()

def get_landmark_coords(row: pd.Series, landmark: str) -> np.ndarray:
    """Extracts (x, y, z) world coordinates for a given landmark from a DataFrame row."""
    base_name = landmark.lower()
    # Assuming column names are like 'wrist_world_x (m)', etc.
    x = row[f"{base_name}_world_x (m)"]
    y = row[f"{base_name}_world_y (m)"]
    z = row[f"{base_name}_world_z (m)"]
    return np.array([x, y, z])

def compute_quaternion(row: pd.Series) -> np.ndarray | None:
    """
    Computes the rotation quaternion (w, x, y, z) that transforms the world frame
    to the hand's local frame.
    """
    try:
        # 1. Get Landmark Coordinates
        O = get_landmark_coords(row, AXES_CONFIG.ORIGIN)
        F_target = get_landmark_coords(row, AXES_CONFIG.FORWARD_TARGET)
        U_hint_target = get_landmark_coords(row, AXES_CONFIG.UP_HINT_TARGET)

        # 2. Define Vectors
        F_vec = F_target - O # Vector pointing forward from the wrist
        U_hint = U_hint_target - O # Vector across the palm from the wrist
        
        # 3. Build Orthonormal Basis (Right-Handed System: X=Forward, Z=Up, Y=Right)
        
        # X-axis (Forward, normalized)
        X_axis = F_vec / np.linalg.norm(F_vec)
        
        # Z-axis (Up/Normal to palm, calculated via cross product, normalized)
        # Note: X x U_hint gives a direction perpendicular to both.
        Z_axis = np.cross(X_axis, U_hint)
        Z_axis = Z_axis / np.linalg.norm(Z_axis)
        
        # Y-axis (Right, guaranteed orthogonal to X and Z, normalized)
        Y_axis = np.cross(Z_axis, X_axis)
        Y_axis = Y_axis / np.linalg.norm(Y_axis)

        # Rotation Matrix (R: Local -> World)
        # R = [X_axis | Y_axis | Z_axis] (each axis is a column vector in world coordinates)
        rotation_matrix = np.column_stack((X_axis, Y_axis, Z_axis))
        
        # 4. Convert Rotation Matrix to Quaternion
        rotation = Rotation.from_matrix(rotation_matrix)
        
        # SciPy's .as_quat() returns (x, y, z, w). We reorder to the common (w, x, y, z) format.
        quat_xyzw = rotation.as_quat()
        quat_wxyz = quat_xyzw[[3, 0, 1, 2]] 
        
        return quat_wxyz
        
    except np.linalg.LinAlgError:
        # This occurs if the vectors F_vec and U_hint are collinear (singular)
        print(f"Warning: Singular matrix error at frame {row['frame_index']} for hand {row['hand_index']}.")
        return None
    except KeyError:
        # This occurs if required landmark columns are missing
        return None

def main_conversion() -> None:
    """Main function to load data, compute quaternions, and save the new CSV."""
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    df_hands = df[df['hand_label'].notna()].copy()
    
    if df_hands.empty:
        print("No valid hand detections with label found. Writing header-only CSV.")
        empty_df = pd.DataFrame(columns=['time_sec', 'frame_index', 'hand_index', 'hand_label', 'hand_score', 'quat_w', 'quat_x', 'quat_y', 'quat_z'])
        empty_df.to_csv(output_quaternion_csv, index=False)
        return

    print(f"Processing {len(df_hands)} hand detections for Quaternion output...")
    
    # Apply the computation function row-wise
    quaternions = df_hands.apply(compute_quaternion, axis=1)
    
    # Filter out rows where quaternion computation failed
    valid_quats = quaternions.dropna()
    valid_indices = valid_quats.index
    
    # Create the quaternion DataFrame
    quat_df = pd.DataFrame(
        valid_quats.tolist(),
        index=valid_indices,
        columns=['quat_w', 'quat_x', 'quat_y', 'quat_z']
    )
    
    # Merge the metadata with the new quaternion data
    core_cols = ['time_sec', 'frame_index', 'hand_index', 'hand_label', 'hand_score']
    result_df = df_hands.loc[valid_indices, core_cols].copy()
    result_df = pd.concat([result_df, quat_df], axis=1)
    
    # Save to CSV
    result_df.to_csv(output_quaternion_csv, index=False)
    
    print(f"✅ Successfully computed quaternions for {len(result_df)} detections.")
    print(f"Quaternions written to {output_quaternion_csv.resolve()}")

if __name__ == "__main__":
    main_conversion()