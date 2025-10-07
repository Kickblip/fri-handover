from __future__ import annotations
################ 3d depth data included for mediapipe landmarks
from pathlib import Path
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from pyk4a import PyK4A, Config, ImageFormat, ColorResolution, DepthMode, FPS

# ==== USER CONFIGURATION ====
input_video = Path("demo-content/output_with_hands.mkv")       # <-- Change this
output_csv = Path("outputs/hand_landmarks_metric.csv")          # <-- Change this
output_video = Path("outputs/output_debug_video_metric.mp4")    # <-- Optional, set to None to skip
max_hands = 2
min_detection_confidence = 0.5
min_tracking_confidence = 0.5
# ============================

_HAND_LANDMARKS = list(mp.solutions.hands.HandLandmark)

def build_columns() -> list[str]:
    """Build column names for the DataFrame including metric coordinates."""
    columns = ["time_sec", "frame_index", "hand_index", "hand_label", "hand_score"]
    for landmark in _HAND_LANDMARKS:
        name = landmark.name.lower()
        # Store both normalized and metric coordinates
        columns.extend([
            f"{name}_norm_x", f"{name}_norm_y", f"{name}_norm_z",  # MediaPipe normalized
            f"{name}_mm_x", f"{name}_mm_y", f"{name}_mm_z"         # Real-world millimeters
        ])
    return columns

def pixel_to_metric(x_px: float, y_px: float, z_depth_mm: float, 
                   fx: float, fy: float, cx: float, cy: float) -> tuple[float, float, float]:
    """Convert pixel coordinates and depth to metric 3D coordinates."""
    # Convert normalized MediaPipe coords [0,1] to pixel coordinates
    x_px = int(x_px * width)
    y_px = int(y_px * height)
    
    # Back-project to 3D using pinhole camera model
    x_mm = (x_px - cx) * z_depth_mm / fx
    y_mm = (y_px - cy) * z_depth_mm / fy
    
    return x_mm, y_mm, z_depth_mm

def main() -> None:
    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    # Initialize Kinect capture
    config = Config(
        color_format=ImageFormat.COLOR_BGRA32,
        color_resolution=ColorResolution.RES_1080P,
        depth_mode=DepthMode.NFOV_UNBINNED,
        synchronized_images_only=True,
        fps=FPS.FPS_30
    )
    kinect = PyK4A(config=config)
    kinect.start()
    
    # Get camera calibration
    calib = kinect.get_calibration()
    camera_matrix = np.array(calib.color_camera_calibration.intrinsics.parameters.param)
    fx, fy = camera_matrix[0], camera_matrix[4]  # Focal lengths
    cx, cy = camera_matrix[2], camera_matrix[5]  # Principal point
    
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=max_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    drawing = mp.solutions.drawing_utils

    rows: list[list[float]] = []
    frame_index = 0

    try:
        while True:
            # Get synchronized color and depth
            capture = kinect.get_capture()
            if capture.color is None or capture.depth is None:
                continue
                
            color = capture.color
            depth = capture.depth
            
            frame_index += 1
            time_sec = frame_index / 30.0  # Kinect fps = 30
            
            # Process with MediaPipe
            rgb_frame = cv2.cvtColor(color, cv2.COLOR_BGRA2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                handedness = results.multi_handedness or []
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Store normalized coords and compute metric coords
                    coords: list[float] = []
                    for landmark in _HAND_LANDMARKS:
                        lm = hand_landmarks.landmark[landmark]
                        
                        # Get depth at landmark position (average small window)
                        x_px = int(lm.x * color.shape[1])
                        y_px = int(lm.y * color.shape[0])
                        z_mm = float(np.median(depth[
                            max(0, y_px-2):min(depth.shape[0], y_px+3),
                            max(0, x_px-2):min(depth.shape[1], x_px+3)
                        ]))
                        
                        # Store normalized MediaPipe coordinates
                        coords.extend([lm.x, lm.y, lm.z])
                        
                        # Convert to metric using depth and intrinsics
                        x_mm, y_mm, z_mm = pixel_to_metric(
                            lm.x, lm.y, z_mm, fx, fy, cx, cy
                        )
                        coords.extend([x_mm, y_mm, z_mm])

                    label = None
                    score = None
                    if idx < len(handedness) and handedness[idx].classification:
                        classification = handedness[idx].classification[0]
                        label = classification.label.lower()
                        score = classification.score

                    rows.append([time_sec, frame_index, idx, label, score] + coords)

                    if output_video:
                        drawing.draw_landmarks(
                            color,
                            hand_landmarks,
                            mp.solutions.hands.HAND_CONNECTIONS,
                            landmark_drawing_spec=drawing.DrawingSpec(
                                color=(0, 255, 0), thickness=2, circle_radius=2
                            ),
                            connection_drawing_spec=drawing.DrawingSpec(
                                color=(60, 180, 75), thickness=2
                            ),
                        )
            
            # Write debug video frame
            if output_video:
                cv2.imwrite(output_video, color)
                
    finally:
        kinect.stop()
        hands.close()

    # Save to CSV
    if rows:
        df = pd.DataFrame(rows, columns=build_columns())
        df.to_csv(output_csv, index=False)
        print(f"Processed {frame_index} frames; stored {len(rows)} hand detections.")
        print(f"Landmarks written to {output_csv.resolve()}")
        
        # Also save pickle for easier loading with types preserved
        df.to_pickle(output_csv.with_suffix('.pkl'))
        print(f"Pickle written to {output_csv.with_suffix('.pkl')}")
    else:
        df = pd.DataFrame(columns=build_columns())
        df.to_csv(output_csv, index=False)
        print("No hand landmarks detected; wrote header-only CSV.")
        print(f"Path: {output_csv.resolve()}")

if __name__ == "__main__":
    main()