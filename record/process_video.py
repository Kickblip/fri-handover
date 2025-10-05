import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

# --- 1. Initialize MediaPipe ---
mp_hands = mp.solutions.hands
# Use the newer, more configurable Hands solution
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- 2. Define Palm/Hand Coordinate System for Orientation ---
# We define the palm's orientation based on a 3-point local coordinate system (LCS)
# This LCS is then compared to a World Coordinate System (WCS) to find the rotation.

# Mediapipe World Landmarks used to define the palm plane:
# P0: WRIST (Landmark 0) -> Origin of the Hand LCS
# P5: INDEX_FINGER_MCP (Landmark 5)
# P17: PINKY_FINGER_MCP (Landmark 17)

# The hand LCS will be:
# Y-axis (Up along the wrist/palm): P0 -> P9 (Middle Finger MCP is often used, but we use an estimate)
# Z-axis (Palm Normal/Direction): Cross product of the other two vectors
# X-axis (Lateral/Side-to-Side): P17 -> P5 (Pinky to Index) or its perpendicular
# To simplify, we calculate the Z-axis (Palm Normal) and convert its rotation from World Z-axis.

def calculate_palm_rotation(world_landmarks):
    """
    Calculates the 3D rotation vector (Rodrigues) for the palm's orientation.
    This is achieved by finding the palm's normal vector and calculating the
    rotation required to move a reference vector (World Z-axis) to this normal.
    """
    if not world_landmarks:
        return np.array([np.nan, np.nan, np.nan])

    # Extract coordinates for the three defining points
    P0_wrist = np.array([world_landmarks.landmark[0].x, world_landmarks.landmark[0].y, world_landmarks.landmark[0].z])
    P5_index_mcp = np.array([world_landmarks.landmark[5].x, world_landmarks.landmark[5].y, world_landmarks.landmark[5].z])
    P17_pinky_mcp = np.array([world_landmarks.landmark[17].x, world_landmarks.landmark[17].y, world_landmarks.landmark[17].z])

    # 1. Define two vectors in the palm plane
    V_index = P5_index_mcp - P0_wrist  # Vector from wrist to index MCP
    V_pinky = P17_pinky_mcp - P0_wrist # Vector from wrist to pinky MCP

    # 2. Calculate the Palm Normal (the Z-axis of the palm's local coordinate system)
    # The cross product gives a vector perpendicular to the plane
    # The magnitude of this vector is the 'direction' of the palm
    palm_normal = np.cross(V_index, V_pinky)
    palm_normal = palm_normal / np.linalg.norm(palm_normal) # Normalize to unit vector
    
    # Check for handedness to ensure the normal points outwards from the palm 
    # (This is a simplified check and may need refinement for complex poses)
    # Based on MediaPipe's hand-ness, a right hand should have a positive V_pinky x V_index normal along the camera Z axis if palm faces camera.
    # The default cross product direction is often defined by the "right-hand rule" for the defined vector order.
    # For a right hand: wrist->index cross wrist->pinky results in a vector pointing *into* the palm.
    # To point *out* of the palm, we reverse the order or multiply by -1
    # For now, we'll assume the simple cross product direction is sufficient for tracking orientation.

    # 3. Calculate the rotation (Rodrigues Vector)
    # The rotation from a reference vector (World Z-axis: [0, 0, 1]) to the calculated palm normal
    # The 'rotation_vector_to_align_vectors' is a helper function that performs this geometric operation.
    
    ref_z_axis = np.array([0.0, 0.0, 1.0])
    
    # Calculate the rotation axis (cross product of reference and target)
    axis = np.cross(ref_z_axis, palm_normal)
    axis_norm = np.linalg.norm(axis)
    
    # If the vectors are perfectly aligned (or opposite), the axis is zero
    if axis_norm < 1e-6:
        # Check if vectors are parallel (0 deg) or anti-parallel (180 deg)
        dot_product = np.dot(ref_z_axis, palm_normal)
        if dot_product > 0.9999:  # Parallel
            return np.array([0.0, 0.0, 0.0])
        else: # Anti-parallel (180 deg rotation around an arbitrary axis, e.g., world X-axis)
            rotation_vector = np.array([np.pi, 0.0, 0.0])
            return rotation_vector

    # Calculate the rotation angle (from dot product)
    angle = np.arccos(np.dot(ref_z_axis, palm_normal))
    
    # Rodrigues vector is: axis * angle
    rotation_vector = (axis / axis_norm) * angle

    return rotation_vector


def process_video(mkv_path, output_csv_path):
    """
    Processes the video, extracts 3D landmarks, calculates Rodrigues rotation, 
    and saves the data to a CSV file.
    """
    cap = cv2.VideoCapture(mkv_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {mkv_path}")
        return

    # Get video properties for time calculation
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    all_landmarks_data = []
    frame_number = 0

    print(f"Processing video: {mkv_path} at {fps:.2f} FPS...")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # Convert the BGR image to RGB for MediaPipe
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False # Read-only for performance

        # Process the image
        results = hands.process(image)
        
        frame_time = frame_number / fps # Calculate time in seconds
        
        # --- Extraction and Calculation ---
        if results.multi_hand_world_landmarks:
            for i, hand_world_landmarks in enumerate(results.multi_hand_world_landmarks):
                # Calculate Palm Orientation (Rodrigues Vector)
                rodrigues_vec = calculate_palm_rotation(hand_world_landmarks)
                
                data = {
                    'frame': frame_number, 
                    'time_sec': frame_time,
                    'hand_idx': i, # For multiple hands, this distinguishes them
                    'RODRIGUES_X': rodrigues_vec[0],
                    'RODRIGUES_Y': rodrigues_vec[1],
                    'RODRIGUES_Z': rodrigues_vec[2]
                }
                
                # Store 3D world coordinates for each of the 21 landmarks
                for j, landmark in enumerate(hand_world_landmarks.landmark):
                    data[f'LM_{j}_X'] = landmark.x
                    data[f'LM_{j}_Y'] = landmark.y
                    data[f'LM_{j}_Z'] = landmark.z
                
                all_landmarks_data.append(data)

        frame_number += 1
        
    cap.release()
    print(f"Processing complete. {frame_number} frames analyzed.")
    
    # --- 4. Store in Pandas DataFrame ---
    df = pd.DataFrame(all_landmarks_data)
    df.to_csv(output_csv_path, index=False)
    print(f"Data saved to {output_csv_path}")
    
    return df

# --- 5. Execution ---
if __name__ == '__main__':
    # IMPORTANT: Change this to the path of your .mkv file
    VIDEO_FILE_PATH = '1_bhavana_final_neel_final.mkv' 
    OUTPUT_FILE_PATH = 'hand_landmarks_and_rotation.csv'

    # Create a dummy .mkv file if you don't have one for testing the script structure
    # You MUST replace 'input_video.mkv' with a valid file for actual data extraction.
    # To run this script fully, you need a proper video file.
    
    # Example usage:
    # df_results = process_video(VIDEO_FILE_PATH, OUTPUT_FILE_PATH)
    # print("\nFirst 5 rows of the generated DataFrame:")
    # print(df_results.head())
    
    print("\n--- Script Setup ---")
    print(f"1. Ensure your video file is named '{VIDEO_FILE_PATH}' or update the path.")
    print(f"2. The output data will be saved to '{OUTPUT_FILE_PATH}'.")
    print("3. Uncomment the 'df_results = process_video...' lines to run the processing.")
    
# Clean up MediaPipe Hands object
hands.close()