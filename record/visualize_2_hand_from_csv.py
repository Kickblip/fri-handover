import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import List, Tuple, Any

# ============================
# ==== CONFIGURATION ====
# ============================
# --- Inputs ---

INPUT_CSV_PATH = Path("outputs/2_handover_cartesian_combined.csv") 
INPUT_VIDEO_PATH  = Path("outputs/2_output_world_coordinates_preview.mp4") 

# --- Outputs ---
OUTPUT_VIDEO_PATH = Path("outputs/handover_visualization_from_csv.mp4")

# --- Parameters ---
MAX_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
# ============================

# MediaPipe Hand Landmarks for Index mapping (0 to 20)
_HAND_LANDMARKS = list(mp.solutions.hands.HandLandmark)

def get_landmark_feature_names() -> List[str]:
    """Generates the list of 63 base coordinate feature names."""
    columns = []
    for landmark in _HAND_LANDMARKS:
        name = landmark.name.lower()
        columns.extend([f"{name}_world_x", f"{name}_world_y", f"{name}_world_z"])
    return columns

def calculate_closest_indices_from_row(row: pd.Series) -> Tuple[int, int]:
    """
    Calculates the indices of the two closest landmarks (0-20) for a single frame (row).
    This logic is repeated here to ensure we find the closest points regardless of 
    what was originally stored in the CSV.
    """
    feature_names = get_landmark_feature_names()
    
    # 1. Extract 3D World Coordinates from the row
    hand0_coords_flat = []
    hand1_coords_flat = []
    
    # Check for NaN/missing data
    hand0_present = not pd.isna(row.get(f"{feature_names[0]}_0", np.nan))
    hand1_present = not pd.isna(row.get(f"{feature_names[0]}_1", np.nan))

    if not (hand0_present and hand1_present):
        return -1, -1 # Indicates one or both hands are missing

    # Extract all 63 coords for each hand, filling any internal NaNs (should be 0 or small)
    for name in feature_names:
        hand0_coords_flat.append(row.get(f"{name}_0", np.nan))
        hand1_coords_flat.append(row.get(f"{name}_1", np.nan))

    # Convert to NumPy arrays (21, 3)
    coords0 = np.array(hand0_coords_flat).reshape(-1, 3)
    coords1 = np.array(hand1_coords_flat).reshape(-1, 3)

    # 2. Vectorized calculation
    # Broadcasting subtraction: (21, 1, 3) - (1, 21, 3) -> (21, 21, 3)
    diff = coords0[:, np.newaxis, :] - coords1[np.newaxis, :, :]
    
    # Calculate squared Euclidean distance: sum(diff^2) over the last axis (x,y,z)
    squared_distances = np.sum(diff**2, axis=2) # Shape: (21, 21)
    
    # 3. Find the minimum distance index
    min_flat_index = np.argmin(squared_distances)
    
    # Convert the flat index back to 2D indices (idx0, idx1)
    idx0, idx1 = np.unravel_index(min_flat_index, squared_distances.shape)
    
    return int(idx0), int(idx1)

def main():
    # --- 1. Load Data ---
    if not INPUT_VIDEO_PATH.exists() or not INPUT_CSV_PATH.exists():
        print("❌ Error: One or both input files not found.")
        print(f"Video: {INPUT_VIDEO_PATH.resolve()}")
        print(f"CSV: {INPUT_CSV_PATH.resolve()}")
        return

    try:
        df = pd.read_csv(INPUT_CSV_PATH)
        df = df.sort_values(by='frame_index').set_index('frame_index')
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return

    cap = cv2.VideoCapture(str(INPUT_VIDEO_PATH))
    if not cap.isOpened():
        print(f"❌ Error: Unable to open video {INPUT_VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # --- Video Writer setup ---
    OUTPUT_VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO_PATH),
        fourcc,
        fps,
        (width, height),
    )

    # --- MediaPipe Hands setup (only for 2D image coordinates) ---
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    drawing = mp.solutions.drawing_utils

    print(f"🎥 Starting video visualization (Total frames: {total_frames})...")
    
    frame_index = 0
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            frame_index += 1
            if frame_index not in df.index:
                print(f"⚠️ Warning: Missing data for frame {frame_index} in CSV. Skipping.")
                writer.write(frame)
                continue
            
            # 1. Get the current row of data from the CSV
            row = df.loc[frame_index]
            
            # 2. Recalculate the closest landmark indices using the 3D data in the CSV
            lm_idx0, lm_idx1 = calculate_closest_indices_from_row(row)

            # --- Check Hand Presence ---
            hand0_is_present = not pd.isna(row.get('hand_score_0', np.nan))
            hand1_is_present = not pd.isna(row.get('hand_score_1', np.nan))

            # 3. Re-run MediaPipe ONLY to get the 2D image coordinates for drawing
            # This is necessary because the normalized 2D coordinates are not in the CSV
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            # Draw landmarks first (optional, but helpful)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                     drawing.draw_landmarks(
                        frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                        connection_drawing_spec=drawing.DrawingSpec(color=(60, 180, 75), thickness=1),
                    )

            # 4. Draw the line if both hands were detected in the current frame AND the indices are valid
            if hand0_is_present and hand1_is_present and lm_idx0 != -1 and results.multi_hand_landmarks:
                
                # Check if MediaPipe detected at least 2 hands this frame
                if len(results.multi_hand_landmarks) >= 2:
                    
                    # Get the normalized 2D landmark objects from the re-detection
                    lm0_norm = results.multi_hand_landmarks[0].landmark[lm_idx0]
                    lm1_norm = results.multi_hand_landmarks[1].landmark[lm_idx1]
                    
                    # Convert normalized coordinates (0 to 1.0) to pixel coordinates
                    pt0 = (int(lm0_norm.x * width), int(lm0_norm.y * height))
                    pt1 = (int(lm1_norm.x * width), int(lm1_norm.y * height))
                    
                    # Draw a distinctive line (e.g., bright red)
                    cv2.line(frame, pt0, pt1, (0, 0, 255), 3) # RED LINE
                    
                    # Draw highlight circles on the two closest points
                    cv2.circle(frame, pt0, 5, (0, 255, 255), -1) # Yellow circle on Hand 0
                    cv2.circle(frame, pt1, 5, (255, 0, 255), -1) # Magenta circle on Hand 1

            # Write the frame to the output video
            writer.write(frame)
                
    finally:
        cap.release()
        hands.close()
        writer.release()
        print(f"\n✅ Visualization video written to {OUTPUT_VIDEO_PATH.resolve()}")

if __name__ == "__main__":
    main()