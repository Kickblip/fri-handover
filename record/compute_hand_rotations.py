from pathlib import Path
import numpy as np
import pandas as pd
import cv2

# MediaPipe hand landmark connection mapping (parent->child relationships)
HAND_CONNECTIONS = {
    # Thumb
    1: 0,  # thumb_cmc -> wrist
    2: 1,  # thumb_mcp -> thumb_cmc
    3: 2,  # thumb_ip -> thumb_mcp
    4: 3,  # thumb_tip -> thumb_ip
    
    # Index finger
    5: 0,  # index_mcp -> wrist
    6: 5,  # index_pip -> index_mcp
    7: 6,  # index_dip -> index_pip
    8: 7,  # index_tip -> index_dip
    
    # Middle finger
    9: 0,   # middle_mcp -> wrist
    10: 9,  # middle_pip -> middle_mcp
    11: 10, # middle_dip -> middle_pip
    12: 11, # middle_tip -> middle_dip
    
    # Ring finger
    13: 0,  # ring_mcp -> wrist
    14: 13, # ring_pip -> ring_mcp
    15: 14, # ring_dip -> ring_pip
    16: 15, # ring_tip -> ring_dip
    
    # Pinky
    17: 0,  # pinky_mcp -> wrist
    18: 17, # pinky_pip -> pinky_mcp
    19: 18, # pinky_dip -> pinky_pip
    20: 19, # pinky_tip -> pinky_dip
}

def compute_direction_vector(point1: np.ndarray, point2: np.ndarray) -> np.ndarray:
    """Compute normalized direction vector from point1 to point2."""
    vector = point2 - point1
    length = np.linalg.norm(vector)
    if length < 1e-6:  # Avoid division by zero
        return np.zeros(3)
    return vector / length

def compute_rodrigues_rotation(direction: np.ndarray, reference: np.ndarray = np.array([0, 0, 1])) -> tuple[np.ndarray, float]:
    """
    Compute Rodrigues rotation vector and angle between direction and reference vector.
    Returns:
        rotation_vector: The axis-angle rotation vector (3D)
        angle: The rotation angle in radians
    """
    # Normalize vectors
    direction = direction / np.linalg.norm(direction)
    reference = reference / np.linalg.norm(reference)
    
    # Compute rotation axis (cross product)
    rotation_axis = np.cross(reference, direction)
    rotation_axis_norm = np.linalg.norm(rotation_axis)
    
    # If vectors are parallel or anti-parallel
    if rotation_axis_norm < 1e-6:
        dot_product = np.dot(reference, direction)
        if dot_product > 0.999:  # Nearly parallel
            return np.zeros(3), 0.0
        else:  # Nearly anti-parallel
            # Choose an arbitrary perpendicular axis
            arbitrary_axis = np.array([1., 0., 0.]) if abs(reference[1]) > 0.1 else np.array([0., 1., 0.])
            rotation_axis = np.cross(reference, arbitrary_axis)
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            return rotation_axis * np.pi, np.pi
    
    # Compute rotation angle
    angle = np.arccos(np.clip(np.dot(reference, direction), -1.0, 1.0))
    
    # Rodrigues rotation vector = axis * angle
    rotation_vector = (rotation_axis / rotation_axis_norm) * angle
    
    return rotation_vector, angle

def process_frame_data(frame_data: pd.DataFrame) -> pd.DataFrame:
    """Process a single frame's hand landmarks to compute rotations."""
    # Initialize storage for computed values
    results = []
    
    # Group by frame and hand
    for (frame, hand_idx), hand_data in frame_data.groupby(['frame_index', 'hand_index']):
        time = hand_data['time_sec'].iloc[0]
        hand_label = hand_data['hand_label'].iloc[0]
        hand_score = hand_data['hand_score'].iloc[0]
        
        # Get 3D positions for all landmarks
        positions = {}
        for i in range(21):  # 21 landmarks
            pos = np.array([
                hand_data[f'landmark_{i}_mm_x'].iloc[0],
                hand_data[f'landmark_{i}_mm_y'].iloc[0],
                hand_data[f'landmark_{i}_mm_z'].iloc[0]
            ])
            positions[i] = pos
            
        # Compute direction vectors and rotations for each landmark
        for landmark_idx in range(21):
            # Get parent landmark
            parent_idx = HAND_CONNECTIONS.get(landmark_idx, 0)  # Default to wrist as parent
            
            # Compute direction vector (child - parent or landmark - wrist)
            direction = compute_direction_vector(
                positions[parent_idx],
                positions[landmark_idx]
            )
            
            # Compute Rodrigues rotation from reference vector
            rotation_vector, angle = compute_rodrigues_rotation(direction)
            
            # Store results
            row = {
                'time_sec': time,
                'frame_index': frame,
                'hand_index': hand_idx,
                'hand_label': hand_label,
                'hand_score': hand_score,
                'landmark_index': landmark_idx,
                # Original positions
                'x_mm': positions[landmark_idx][0],
                'y_mm': positions[landmark_idx][1],
                'z_mm': positions[landmark_idx][2],
                # Direction vector
                'dir_x': direction[0],
                'dir_y': direction[1],
                'dir_z': direction[2],
                # Rotation vector (axis-angle representation)
                'rot_x': rotation_vector[0],
                'rot_y': rotation_vector[1],
                'rot_z': rotation_vector[2],
                'rot_angle': angle,  # in radians
            }
            results.append(row)
    
    return pd.DataFrame(results)

def main():
    # Input/Output paths
    input_csv = Path("outputs/hand_landmarks_metric.csv")
    output_csv = Path("outputs/hand_landmarks_with_rotations.csv")
    
    print(f"Reading landmarks from {input_csv}")
    df = pd.read_csv(input_csv)
    
    print("Computing direction vectors and rotations...")
    results_df = process_frame_data(df)
    
    # Save results
    results_df.to_csv(output_csv, index=False)
    results_df.to_pickle(output_csv.with_suffix('.pkl'))
    
    print(f"\nResults saved to:")
    print(f"CSV: {output_csv}")
    print(f"Pickle: {output_csv.with_suffix('.pkl')}")
    
    # Print some statistics
    print(f"\nProcessed:")
    print(f"- {len(df['frame_index'].unique())} frames")
    print(f"- {len(df['hand_index'].unique())} unique hands")
    print(f"- {len(results_df)} total landmark measurements")
    
    # Example of data for first landmark
    print("\nExample data for first landmark:")
    example = results_df.iloc[0]
    print(f"Position (mm): ({example.x_mm:.1f}, {example.y_mm:.1f}, {example.z_mm:.1f})")
    print(f"Direction: ({example.dir_x:.3f}, {example.dir_y:.3f}, {example.dir_z:.3f})")
    print(f"Rotation (axis-angle): ({example.rot_x:.3f}, {example.rot_y:.3f}, {example.rot_z:.3f})")
    print(f"Rotation angle: {np.degrees(example.rot_angle):.1f} degrees")

if __name__ == "__main__":
    main()