import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import List, Tuple, Any, Optional

# ============================ 
# ==== CONFIGURATION ====
# ============================
# --- Inputs ---
INPUT_CSV_PATH = Path("dataset/mediapipe_outputs/csv/2_w_b.csv") 
INPUT_VIDEO_PATH  = Path("dataset/input_video/2_w_b.mkv")  

# --- Outputs ---
OUTPUT_VIDEO_PATH = Path("dataset/mediapipe_outputs/video/2_w_b_2hands_visualize_closest_pairs.mp4")
OUTPUT_LINES_CSV_PATH = Path("dataset/mediapipe_outputs/csv/2_w_b_2hands_all_pairs.csv")  # per-frame closest pair (if drawn)
OUTPUT_GLOBAL_MIN_CSV_PATH = Path("dataset/mediapipe_outputs/csv/2_w_b_closest_pair.csv")  # NEW: single row (global closest)

# --- Parameters ---
MAX_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
# ============================

# MediaPipe Hand Landmarks for Index mapping (0 to 20)
_HAND_LANDMARKS = list(mp.solutions.hands.HandLandmark)

def get_landmark_feature_names() -> List[str]:
    """Generates the list of 63 base coordinate feature names (world coords)."""
    columns = []
    for landmark in _HAND_LANDMARKS:
        name = landmark.name.lower()
        columns.extend([f"{name}_world_x", f"{name}_world_y", f"{name}_world_z"])
    return columns

def lm_name(idx: int) -> str:
    return _HAND_LANDMARKS[idx].name.lower()

def row_has_hand(row: pd.Series, feature_names: List[str], hand_idx: int) -> bool:
    """Checks if the given hand's first coordinate exists for this row."""
    first_feat = f"{feature_names[0]}_{hand_idx}"
    return not pd.isna(row.get(first_feat, np.nan))

def extract_world_coords_from_row(row: pd.Series, landmark_idx: int, hand_idx: int) -> Optional[Tuple[float, float, float]]:
    """
    Extracts (x,y,z) world coordinates for a single landmark & hand from the CSV row.
    Returns None if any component is missing.
    """
    name = lm_name(landmark_idx)
    x = row.get(f"{name}_world_x_{hand_idx}", np.nan)
    y = row.get(f"{name}_world_y_{hand_idx}", np.nan)
    z = row.get(f"{name}_world_z_{hand_idx}", np.nan)
    if pd.isna(x) or pd.isna(y) or pd.isna(z):
        return None
    return float(x), float(y), float(z)

def calculate_closest_indices_from_row(row: pd.Series) -> Tuple[int, int, float]:
    """
    Calculates the indices of the two closest landmarks (0-20) for a single frame (row),
    using 3D world coordinates from the CSV. Returns (idx0, idx1, min_distance).
    If either hand is missing, returns (-1, -1, np.inf).
    """
    feature_names = get_landmark_feature_names()

    # Check for hand presence
    if not (row_has_hand(row, feature_names, 0) and row_has_hand(row, feature_names, 1)):
        return -1, -1, np.inf

    # Extract all 63 coords for each hand
    hand0_coords_flat = []
    hand1_coords_flat = []
    for name in feature_names:
        hand0_coords_flat.append(row.get(f"{name}_0", np.nan))
        hand1_coords_flat.append(row.get(f"{name}_1", np.nan))

    coords0 = np.array(hand0_coords_flat, dtype=float).reshape(-1, 3)  # (21,3)
    coords1 = np.array(hand1_coords_flat, dtype=float).reshape(-1, 3)  # (21,3)

    # If any landmark missing -> bail
    if np.isnan(coords0).any() or np.isnan(coords1).any():
        return -1, -1, np.inf

    # Vectorized pairwise squared distances (21x21)
    diff = coords0[:, None, :] - coords1[None, :, :]
    squared_distances = np.sum(diff**2, axis=2)
    min_flat_index = np.argmin(squared_distances)
    idx0, idx1 = np.unravel_index(min_flat_index, squared_distances.shape)
    min_distance = float(np.sqrt(squared_distances[idx0, idx1]))

    return int(idx0), int(idx1), min_distance

def main():
    # --- 1. Load Data ---
    if not INPUT_VIDEO_PATH.exists() or not INPUT_CSV_PATH.exists():
        print("❌ Error: One or both input files not found.")
        print(f"Video: {INPUT_VIDEO_PATH.resolve()}")
        print(f"CSV: {INPUT_CSV_PATH.resolve()}")
        return

    try:
        df = pd.read_csv(INPUT_CSV_PATH)
        # Expect the CSV to have a 'frame_index' column
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

    # === Logging containers ===
    line_events = []  # Each entry is a dict captured when line is drawn
    global_min = {
        "distance": np.inf,
        "frame_index": -1,
        "lm_idx0": -1,
        "lm_idx1": -1,
        "w0": None,
        "w1": None,
        "pt0": None,
        "pt1": None,
    }

    feature_names = get_landmark_feature_names()
    frame_index = 0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_index += 1
            if frame_index not in df.index:
                # no data for this frame
                writer.write(frame)
                continue

            # 1) current row
            row = df.loc[frame_index]

            # 2) closest landmark indices + 3D distance
            lm_idx0, lm_idx1, min_dist_world = calculate_closest_indices_from_row(row)

            # 3) Re-run MediaPipe for 2D image coords for drawing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            # Draw landmarks (optional)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                        connection_drawing_spec=drawing.DrawingSpec(color=(60, 180, 75), thickness=1),
                    )

            drew_line = False
            pt0 = pt1 = None

            # 4) Draw line & log the event if valid
            if (lm_idx0 != -1 and lm_idx1 != -1
                and results.multi_hand_landmarks
                and len(results.multi_hand_landmarks) >= 2):
                # Normalized 2D points from MediaPipe
                lm0_norm = results.multi_hand_landmarks[0].landmark[lm_idx0]
                lm1_norm = results.multi_hand_landmarks[1].landmark[lm_idx1]

                pt0 = (int(lm0_norm.x * width), int(lm0_norm.y * height))
                pt1 = (int(lm1_norm.x * width), int(lm1_norm.y * height))

                # Draw a distinctive line
                cv2.line(frame, pt0, pt1, (0, 0, 255), 3)  # red line
                cv2.circle(frame, pt0, 5, (0, 255, 255), -1)  # yellow
                cv2.circle(frame, pt1, 5, (255, 0, 255), -1)  # magenta
                drew_line = True

            # If we drew the line, collect full event info
            if drew_line:
                w0 = extract_world_coords_from_row(row, lm_idx0, 0)
                w1 = extract_world_coords_from_row(row, lm_idx1, 1)

                # Fallback: if world coords missing here for some reason, recompute distance=None
                if (w0 is None) or (w1 is None):
                    this_dist = None
                else:
                    this_dist = float(np.linalg.norm(np.array(w0) - np.array(w1)))

                line_events.append({
                    "frame_index": frame_index,
                    "lm_idx0": lm_idx0,
                    "lm_idx1": lm_idx1,
                    "lm_name0": lm_name(lm_idx0),
                    "lm_name1": lm_name(lm_idx1),
                    "world_distance": this_dist if this_dist is not None else min_dist_world,
                    "pt0_x": pt0[0] if pt0 else None,
                    "pt0_y": pt0[1] if pt0 else None,
                    "pt1_x": pt1[0] if pt1 else None,
                    "pt1_y": pt1[1] if pt1 else None,
                    "w0_x": w0[0] if w0 else None,
                    "w0_y": w0[1] if w0 else None,
                    "w0_z": w0[2] if w0 else None,
                    "w1_x": w1[0] if w1 else None,
                    "w1_y": w1[1] if w1 else None,
                    "w1_z": w1[2] if w1 else None,
                })

                # Track global minimum (using world distance)
                effective_dist = line_events[-1]["world_distance"]
                if effective_dist is not None and effective_dist < global_min["distance"]:
                    global_min.update({
                        "distance": effective_dist,
                        "frame_index": frame_index,
                        "lm_idx0": lm_idx0,
                        "lm_idx1": lm_idx1,
                        "w0": w0,
                        "w1": w1,
                        "pt0": pt0,
                        "pt1": pt1,
                    })

            # If this is the global min frame, annotate it specially
            if frame_index == global_min["frame_index"]:
                cv2.putText(frame, "GLOBAL MIN", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)

            # Write the frame to the output video
            writer.write(frame)

    finally:
        cap.release()
        hands.close()
        writer.release()

        # Save the per-frame closest-pair CSV
        if line_events:
            OUTPUT_LINES_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(line_events).to_csv(OUTPUT_LINES_CSV_PATH, index=False)
            print(f"📝 Logged {len(line_events)} line-drawing events to {OUTPUT_LINES_CSV_PATH.resolve()}")
        else:
            print("ℹ️ No line-drawing events were logged (check detections/CSV alignment).")

        # Save the single-row GLOBAL MIN CSV
        if np.isfinite(global_min["distance"]):
            gm_row = {
                "frame_index": global_min["frame_index"],
                "lm_idx0": global_min["lm_idx0"],
                "lm_idx1": global_min["lm_idx1"],
                "lm_name0": lm_name(global_min["lm_idx0"]),
                "lm_name1": lm_name(global_min["lm_idx1"]),
                "world_distance": global_min["distance"],
                "pt0_x": global_min["pt0"][0] if global_min["pt0"] else None,
                "pt0_y": global_min["pt0"][1] if global_min["pt0"] else None,
                "pt1_x": global_min["pt1"][0] if global_min["pt1"] else None,
                "pt1_y": global_min["pt1"][1] if global_min["pt1"] else None,
                "w0_x": global_min["w0"][0] if global_min["w0"] else None,
                "w0_y": global_min["w0"][1] if global_min["w0"] else None,
                "w0_z": global_min["w0"][2] if global_min["w0"] else None,
                "w1_x": global_min["w1"][0] if global_min["w1"] else None,
                "w1_y": global_min["w1"][1] if global_min["w1"] else None,
                "w1_z": global_min["w1"][2] if global_min["w1"] else None,
            }
            OUTPUT_GLOBAL_MIN_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([gm_row]).to_csv(OUTPUT_GLOBAL_MIN_CSV_PATH, index=False)
            print(f"🏁 Global-min pair saved to {OUTPUT_GLOBAL_MIN_CSV_PATH.resolve()}")
        else:
            print("\nℹ️ Could not determine a global minimum (no valid pairs found).")

        print(f"\n✅ Visualization video written to {OUTPUT_VIDEO_PATH.resolve()}")

if __name__ == "__main__":
    main()