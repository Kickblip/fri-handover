from __future__ import annotations

from pathlib import Path
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

# ============================
# ==== USER CONFIGURATION ====
# ============================
# Input/Output Paths
input_video = Path("demo-content/output_with_hands.mp4")      # <-- Change this to your video path
output_csv = Path("outputs/handover_cartesian_combined.csv") # <-- Final CSV with one row per frame
output_video = Path("outputs/output_world_coordinates_preview.mp4") # <-- Optional preview video
max_hands = 2 # Crucial: Allows detection of two hands
min_detection_confidence = 0.5
min_tracking_confidence = 0.5
# ============================

# MediaPipe Hand Landmarks
_HAND_LANDMARKS = list(mp.solutions.hands.HandLandmark)

def get_landmark_feature_names() -> List[str]:
    """
    Generates the list of 63 base coordinate feature names without hand suffixes.
    Example: ['wrist_world_x', 'wrist_world_y', 'wrist_world_z', ...]
    """
    columns = []
    for landmark in _HAND_LANDMARKS:
        name = landmark.name.lower()
        # Use simple, clean names, omitting the ' (m)' unit marker here for simplicity
        # Unit is implied by 'world' and the output DataFrame documentation
        columns.extend([f"{name}_world_x", f"{name}_world_y", f"{name}_world_z"])
    return columns

def get_final_columns() -> List[str]:
    """
    Generates the full list of columns for the final, pivoted CSV.
    """
    base_cols = ["time_sec", "frame_index"]
    
    # Metadata for each hand
    meta_cols = [f"hand_label_{i}" for i in range(max_hands)] + \
                [f"hand_score_{i}" for i in range(max_hands)]

    # Coordinate columns for each hand (126 columns total for 2 hands)
    feature_names = get_landmark_feature_names()
    coord_cols = []
    for i in range(max_hands):
        coord_cols.extend([f"{name}_{i}" for name in feature_names])
    
    return base_cols + meta_cols + coord_cols

def extract_coords_and_meta(
    results: Any, 
    idx: int
) -> tuple[List[float], Optional[str], Optional[float]]:
    """Helper to extract coordinates, label, and score for a single hand."""
    
    coords: List[float] = []
    
    # Extract Coordinates
    if results.multi_hand_world_landmarks:
        world_landmarks = results.multi_hand_world_landmarks[idx]
        for landmark in _HAND_LANDMARKS:
            lm = world_landmarks.landmark[landmark]
            # World coordinates in meters
            coords.extend([lm.x, lm.y, lm.z])

    # Extract Handedness (Label and Score)
    label = None
    score = None
    if results.multi_handedness and idx < len(results.multi_handedness):
        handedness = results.multi_handedness[idx]
        if handedness.classification:
            classification = handedness.classification[0]
            label = classification.label.lower()
            score = classification.score
            
    return coords, label, score


def main() -> None:
    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Video Writer setup
    write_video = output_video is not None
    if write_video:
        out_path = output_video.with_suffix(".mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(out_path),
            fourcc,
            fps if fps > 0 else 30.0,
            (width, height),
        )
    else:
        writer = None

    # MediaPipe Hands setup
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=max_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    drawing = mp.solutions.drawing_utils

    rows: List[Dict[str, Any]] = []
    frame_index = 0
    feature_names = get_landmark_feature_names()
    final_columns = get_final_columns()

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            frame_index += 1
            time_sec = (frame_index - 1) / fps if fps > 0 else 0.0

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            # Initialize a dictionary for the single output row for this frame
            row_data: Dict[str, Any] = {
                "time_sec": time_sec, 
                "frame_index": frame_index
            }
            
            # Populate metadata and coordinates for ALL expected hands (0 and 1)
            detected_indices = set()
            num_detected = len(results.multi_hand_world_landmarks) if results.multi_hand_world_landmarks else 0

            if num_detected > 0:
                for idx in range(num_detected):
                    # Step 1: Extract data for the current hand (idx)
                    coords, label, score = extract_coords_and_meta(results, idx)
                    
                    # Step 2: Store data in the row dictionary with the correct suffix (_idx)
                    if idx < max_hands: # Ensure we only process up to max_hands
                        # Store metadata
                        row_data[f"hand_label_{idx}"] = label
                        row_data[f"hand_score_{idx}"] = score
                        
                        # Store coordinates
                        for feature_name, coord_value in zip(feature_names, coords):
                            row_data[f"{feature_name}_{idx}"] = coord_value
                        
                        detected_indices.add(idx)

            # Step 3: Fill missing hand data with NaN (or None)
            for i in range(max_hands):
                if i not in detected_indices:
                    # Fill metadata for missing hand
                    row_data[f"hand_label_{i}"] = np.nan
                    row_data[f"hand_score_{i}"] = np.nan
                    
                    # Fill coordinate data for missing hand (all 63 coords)
                    for feature_name in feature_names:
                        row_data[f"{feature_name}_{i}"] = np.nan

            rows.append(row_data)

            # Drawing (uses image coordinates)
            if writer is not None and results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=drawing.DrawingSpec(
                            color=(0, 255, 0), thickness=2, circle_radius=2
                        ),
                        connection_drawing_spec=drawing.DrawingSpec(
                            color=(60, 180, 75), thickness=2
                        ),
                    )
            
            if writer is not None:
                writer.write(frame)
                
    finally:
        cap.release()
        hands.close()
        if writer is not None:
            writer.release()

    # Save to CSV
    if rows:
        # Use the pre-defined columns to ensure order and completeness
        df = pd.DataFrame(rows, columns=final_columns)
        
        # Add the unit marker back to the coordinate columns if desired, 
        # or leave them clean as features (clean is better for ML)
        print("NOTE: Coordinates are world coordinates in meters (m).")
        df.to_csv(output_csv, index=False)
        
        print(f"\n✅ Processing complete. {frame_index} total frames analyzed.")
        print(f"✅ Stored {len(rows)} frames of combined two-hand data.")
        print(f"✅ Combined world Cartesian data written to {output_csv.resolve()}")
    else:
        df = pd.DataFrame(columns=final_columns)
        df.to_csv(output_csv, index=False)
        print("No hand landmarks detected; wrote header-only CSV.")

    if write_video:
        print(f"Preview video written to {out_path.resolve()}")


if __name__ == "__main__":
    main()
