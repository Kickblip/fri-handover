from pathlib import Path
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

# ==== CONFIGURATION ====
input_csv = Path("outputs/hand_world_landmarks_with_quaternion.csv")
output_csv = Path("outputs/hand_world_landmarks_with_rodrigues.csv")
# ========================


def quaternion_to_rodrigues(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts hand orientation data from Quaternions [w, x, y, z] to
    Rodrigues Rotation Vectors [r_x, r_y, r_z].
    
    The quaternion data must be in the format: w, x, y, z.
    SciPy's 'from_quat' expects scalar-last (x, y, z, w), so we must rearrange it.
    """
    
    # 1. Prepare Quaternion Data
    # The original CSV uses [w, x, y, z]. SciPy's R.from_quat expects [x, y, z, w].
    quat_cols = ['hand_rot_w', 'hand_rot_x', 'hand_rot_y', 'hand_rot_z']
    
    # Select and reorder the columns to [x, y, z, w] for SciPy
    quaternion_data = df[['hand_rot_x', 'hand_rot_y', 'hand_rot_z', 'hand_rot_w']].values
    
    # Handle NaN values (where hand detection or quaternion calculation failed)
    # We'll calculate rotations only for valid rows and fill NaNs later
    valid_indices = ~np.isnan(quaternion_data).any(axis=1)
    valid_quats = quaternion_data[valid_indices]
    
    if len(valid_quats) == 0:
        print("Warning: No valid quaternion data found for conversion.")
        
        # Create empty columns for the rotation vector
        df['hand_rotvec_x'] = np.nan
        df['hand_rotvec_y'] = np.nan
        df['hand_rotvec_z'] = np.nan
        return df

    # 2. Perform the Conversion using SciPy
    # R.from_quat converts the [x, y, z, w] array into a Rotation object
    # .as_rotvec() converts the Rotation object into the Rodrigues Vector (Axis-Angle) [r_x, r_y, r_z]
    try:
        rotation_objects = R.from_quat(valid_quats)
        rodrigues_vectors = rotation_objects.as_rotvec() # [r_x, r_y, r_z]
    except Exception as e:
        print(f"Error during rotation conversion: {e}")
        # Fallback to NaN
        rodrigues_vectors = np.full((len(valid_quats), 3), np.nan)


    # 3. Insert results back into a DataFrame
    # Create an empty array to hold the new rotation vectors, size matches the original DataFrame
    full_rodrigues = np.full((len(df), 3), np.nan)
    
    # Insert the calculated vectors back at their original valid locations
    full_rodrigues[valid_indices] = rodrigues_vectors
    
    # Create new columns for the Rodrigues vector
    df['hand_rotvec_x'] = full_rodrigues[:, 0]
    df['hand_rotvec_y'] = full_rodrigues[:, 1]
    df['hand_rotvec_z'] = full_rodrigues[:, 2]
    
    return df


def main() -> None:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}. Please run the first script first.")

    print(f"Loading data from: {input_csv}")
    df = pd.read_csv(input_csv)

    # Check for the required quaternion columns
    required_cols = ['hand_rot_w', 'hand_rot_x', 'hand_rot_y', 'hand_rot_z']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(
            f"Input CSV must contain quaternion columns: {required_cols}. "
            "Ensure the previous script ran successfully."
        )

    # Convert the quaternion columns to Rodrigues vectors
    df_result = quaternion_to_rodrigues(df)

    # Save the updated DataFrame to a new CSV
    df_result.to_csv(output_csv, index=False)
    
    print("-" * 50)
    print(f"Conversion complete!")
    print(f"Total rows processed: {len(df)}")
    print(f"Output saved to: {output_csv.resolve()}")
    print(f"New columns added: ['hand_rotvec_x', 'hand_rotvec_y', 'hand_rotvec_z']")
    print("-" * 50)


if __name__ == "__main__":
    main()