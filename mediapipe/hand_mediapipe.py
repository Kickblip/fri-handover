from __future__ import annotations

from pathlib import Path
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

# 🔹 NEW: Azure Kinect imports
try:
    from pyk4a import PyK4A, Config, CalibrationType
except ImportError:
    raise ImportError(
        "pyk4a is required for this script but is not installed.\n"
        "To install pyk4a, you need Microsoft Visual C++ Build Tools.\n"
        "1. Download and install: https://visualstudio.microsoft.com/visual-cpp-build-tools/\n"
        "2. Then run: pip install pyk4a\n"
        "Alternatively, use '2d_hand_mediapipe.py' if you don't need depth information."
    )

# ============================ 
# ==== USER CONFIGURATION ====
# ============================
# Input/Output Paths
# input_video = Path("dataset/input_video/name_of_input.mkv")
# input_video = Path("dataset/mediapipe_ouputs/video/world/1_w_b_world.mp4") ???
# output_csv = Path("dataset/mediapipe_outputs/csv/world/name_of_file_extracted_from_input_video_world.csv")
# output_video = Path("dataset/mediapipe_outputs/video/world/name_of_file_extracted_from_input_video_world.mp4")

max_hands = 2
min_detection_confidence = 0.5
min_tracking_confidence = 0.5
# ============================

mp_hands = mp.solutions.hands
_HAND_LANDMARKS = list(mp_hands.HandLandmark)


# ====================================================
# ============ HELPER FUNCTION DEFINITIONS ============
# ====================================================
def get_landmark_feature_names() -> List[str]:
    columns = []
    for landmark in _HAND_LANDMARKS:
        name = landmark.name.lower()
        columns.extend([
            f"{name}_norm_x", f"{name}_norm_y",
            f"{name}_px_x", f"{name}_px_y",
            f"{name}_depth_m", f"{name}_camX", f"{name}_camY", f"{name}_camZ"
        ])
    return columns


def get_final_columns() -> List[str]:
    base_cols = ["time_sec", "frame_index"]
    meta_cols = [f"hand_label_{i}" for i in range(max_hands)] + \
                [f"hand_score_{i}" for i in range(max_hands)]
    feature_names = get_landmark_feature_names()
    coord_cols = []
    for i in range(max_hands):
        coord_cols.extend([f"{name}_{i}" for name in feature_names])
    return base_cols + meta_cols + coord_cols


def extract_landmarks_with_depth(
    results, depth_frame, intrinsics, width, height
):
    """
    Extract normalized (x, y), pixel (x, y), depth, and 3D camera-space coordinates
    for each detected hand.
    """
    hand_data = []
    if not results.multi_hand_landmarks:
        return hand_data

    fx, fy, cx, cy = (
        intrinsics[0, 0],
        intrinsics[1, 1],
        intrinsics[0, 2],
        intrinsics[1, 2],
    )

    for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
        coords = []
        for lm in _HAND_LANDMARKS:
            lm_obj = hand_landmarks.landmark[lm]

            # Normalized coordinates
            x_norm = lm_obj.x
            y_norm = lm_obj.y

            # Convert to pixel coordinates
            x_px = int(x_norm * width)
            y_px = int(y_norm * height)

            # Clamp indices to avoid out-of-bounds
            x_px = np.clip(x_px, 0, width - 1)
            y_px = np.clip(y_px, 0, height - 1)

            # Depth in millimeters → meters
            depth_m = depth_frame[y_px, x_px] / 1000.0 if depth_frame is not None else np.nan

            # Convert to camera 3D (relative coordinates)
            if not np.isnan(depth_m):
                X = (x_px - cx) * depth_m / fx
                Y = (y_px - cy) * depth_m / fy
                Z = depth_m
            else:
                X = Y = Z = np.nan

            coords.extend([x_norm, y_norm, x_px, y_px, depth_m, X, Y, Z])
        hand_data.append(coords)
    return hand_data


# ====================================================
# ===================== MAIN LOOP =====================
# ====================================================
def main() -> None:
    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 🔹 Initialize Azure Kinect
    print("Initializing Azure Kinect...")
    k4a = PyK4A(Config(color_resolution=720, depth_mode=2))
    k4a.start()
    intrinsics = k4a.calibration.get_camera_matrix(CalibrationType.COLOR)
    print("Azure Kinect initialized successfully.")

    # Video Writer setup
    write_video = output_video is not None
    if write_video:
        out_path = output_video.with_suffix(".mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps if fps > 0 else 30.0, (width, height))
    else:
        writer = None

    hands = mp_hands.Hands(
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

            # 🔹 Get synchronized depth frame from Kinect
            capture = k4a.get_capture()
            depth_frame = capture.transformed_depth  # aligned with color image

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            row_data: Dict[str, Any] = {"time_sec": time_sec, "frame_index": frame_index}
            detected_indices = set()
            num_detected = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0

            if num_detected > 0:
                all_hand_data = extract_landmarks_with_depth(results, depth_frame, intrinsics, width, height)

                for idx in range(num_detected):
                    if idx < max_hands:
                        row_data[f"hand_label_{idx}"] = (
                            results.multi_handedness[idx].classification[0].label.lower()
                            if results.multi_handedness
                            else np.nan
                        )
                        row_data[f"hand_score_{idx}"] = (
                            results.multi_handedness[idx].classification[0].score
                            if results.multi_handedness
                            else np.nan
                        )

                        for feature_name, coord_value in zip(feature_names, all_hand_data[idx]):
                            row_data[f"{feature_name}_{idx}"] = coord_value

                        detected_indices.add(idx)

            # Fill missing hands
            for i in range(max_hands):
                if i not in detected_indices:
                    row_data[f"hand_label_{i}"] = np.nan
                    row_data[f"hand_score_{i}"] = np.nan
                    for feature_name in feature_names:
                        row_data[f"{feature_name}_{i}"] = np.nan

            rows.append(row_data)

            if writer is not None and results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        drawing.DrawingSpec(color=(60, 180, 75), thickness=2),
                    )
                writer.write(frame)

    finally:
        cap.release()
        k4a.stop()
        hands.close()
        if writer is not None:
            writer.release()

    # Save results
    df = pd.DataFrame(rows, columns=final_columns)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Done! Saved {len(rows)} frames with depth and 3D coordinates → {output_csv}")


if __name__ == "__main__":
    main()
