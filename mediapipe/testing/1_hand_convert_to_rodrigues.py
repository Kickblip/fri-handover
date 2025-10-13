from pathlib import Path
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation
from typing import NamedTuple, Literal

# ==== USER CONFIGURATION FOR RODRIGUES SCRIPT ====
# NOTE: Uses the same input file as the quaternion script.
input_csv = Path("dataset/mediapipe_outputs/csv/2_w_b.csv")
output_rodrigues_csv = Path("dataset/mediapipe_outputs/csv/1hand_rodrigues.csv") # <-- Target output file for Rodrigues
# ==========================================

# Define the landmarks to use for the local coordinate system
class HandAxes(NamedTuple):
    """Defines the hand landmarks used to construct the local coordinate system."""
    ORIGIN: Literal['WRIST'] = 'WRIST'
    FORWARD_TARGET: Literal['MIDDLE_FINGER_MCP'] = 'MIDDLE_FINGER_MCP'
    UP_HINT_TARGET: Literal['INDEX_FINGER_MCP'] = 'INDEX_FINGER_MCP'

AXES_CONFIG = HandAxes()

def get_landmark_coords(row: pd.Series, landmark: str) -> np.ndarray:
    """Extracts (x, y, z) world coordinates for a given landmark from a DataFrame row."""
    base_name = landmark.lower()
    x = row[f"{base_name}_world_x (m)"]
    y = row[f"{base_name}_world_y (m)"]
    z = row[f"{base_name}_world_z (m)"]
    return np.array([x, y, z])

def compute_rodrigues_vector(row: pd.Series) -> np.ndarray | None:
    """
    Computes the Rodrigues rotation vector (x, y, z) that transforms the world frame
    to the hand's local frame.
    """
    try:
        # 1. Get Landmark Coordinates
        O = get_landmark_coords(row, AXES_CONFIG.ORIGIN)
        F_target = get_landmark_coords(row, AXES_CONFIG.FORWARD_TARGET)
        U_hint_target = get_landmark_coords(row, AXES_CONFIG.UP_HINT_TARGET)

        # 2. Define Vectors
        F_vec = F_target - O
        U_hint = U_hint_target - O
        
        # 3. Build Orthonormal Basis (Right-Handed System: X=Forward, Z=Up, Y=Right)
        X_axis = F_vec / np.linalg.norm(F_vec)
        
        # Z-axis (Up)
        Z_axis = np.cross(X_axis, U_hint)
        Z_axis = Z_axis / np.linalg.norm(Z_axis)
        
        # Y-axis (Right)
        Y_axis = np.cross(Z_axis, X_axis)
        Y_axis = Y_axis / np.linalg.norm(Y_axis)

        # Rotation matrix (R: Local -> World)
        rotation_matrix = np.column_stack((X_axis, Y_axis, Z_axis))
        
        # 4. Convert Rotation Matrix to SciPy Rotation Object
        rotation = Rotation.from_matrix(rotation_matrix)
        
        # 5. Extract Rodrigues Vector (Rotation Vector)
        # The vector is (x, y, z), where its magnitude is the rotation angle in radians.
        rot_vec_xyz = rotation.as_rotvec() 
        
        return rot_vec_xyz
        
    except np.linalg.LinAlgError:
        print(f"Warning: Singular matrix encountered at frame {row['frame_index']} for hand {row['hand_index']}. Skipping.")
        return None
    except KeyError:
        print(f"Warning: Missing landmark data at frame {row['frame_index']} for hand {row['hand_index']}. Skipping.")
        return None

def main_conversion_rodrigues() -> None:
    """Main function to load data, compute Rodrigues vectors, and save the new CSV."""
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    df_hands = df[df['hand_label'].notna()].copy()
    
    if df_hands.empty:
        print("No valid hand detections with label found in the input CSV. Outputting header-only CSV.")
        empty_df = pd.DataFrame(columns=['time_sec', 'frame_index', 'hand_index', 'hand_label', 'hand_score', 'rot_vec_x', 'rot_vec_y', 'rot_vec_z'])
        empty_df.to_csv(output_rodrigues_csv, index=False)
        return

    print(f"Processing {len(df_hands)} hand detections for Rodrigues Vector output...")
    
    # Apply the computation function row-wise
    rot_vectors = df_hands.apply(compute_rodrigues_vector, axis=1)
    
    # Filter out rows where computation failed
    valid_vectors = rot_vectors.dropna()
    valid_indices = valid_vectors.index
    
    # Create the Rodrigues Vector DataFrame
    rot_vec_df = pd.DataFrame(
        valid_vectors.tolist(),
        index=valid_indices,
        columns=['rot_vec_x', 'rot_vec_y', 'rot_vec_z']
    )
    
    # Select the core columns we want to keep
    core_cols = ['time_sec', 'frame_index', 'hand_index', 'hand_label', 'hand_score']
    result_df = df_hands.loc[valid_indices, core_cols].copy()
    
    # Merge the core data with the new Rodrigues vector data
    result_df = pd.concat([result_df, rot_vec_df], axis=1)
    
    # Save to CSV
    result_df.to_csv(output_rodrigues_csv, index=False)
    
    print(f"✅ Successfully computed Rodrigues Vectors for {len(result_df)} detections.")
    print(f"Results written to {output_rodrigues_csv.resolve()}")


if __name__ == "__main__":
    main_conversion_rodrigues()