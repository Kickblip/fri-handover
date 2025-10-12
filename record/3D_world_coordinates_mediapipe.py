from __future__ import annotations

from pathlib import Path
import cv2
import mediapipe as mp
import pandas as pd

### ONE HAND
# ==== USER CONFIGURATION ====
input_video = Path("1_bhavana_final_neel_final.mkv")       # <-- Change this
output_csv = Path("outputs/hand_world_landmarks.csv")          # <-- Renamed for clarity
output_video = Path("outputs/output_world_coordinates.mp4")            # <-- Optional, set to None to skip
max_hands = 2
min_detection_confidence = 0.5
min_tracking_confidence = 0.5
# ============================


_HAND_LANDMARKS = list(mp.solutions.hands.HandLandmark)


def build_columns() -> list[str]:
    # --- CORRECTED: Use 'world' in column names ---
    columns = ["time_sec", "frame_index", "hand_index", "hand_label", "hand_score"]
    for landmark in _HAND_LANDMARKS:
        name = landmark.name.lower()
        # Indicate that the coordinates are 3D World Coordinates in meters
        columns.extend([f"{name}_world_x (m)", f"{name}_world_y (m)", f"{name}_world_z (m)"])
    return columns


def main() -> None:
    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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
            success, frame = cap.read()
            if not success:
                break
            frame_index += 1
            time_sec = (frame_index - 1) / fps if fps > 0 else 0.0

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            # NOTE: multi_hand_landmarks is used for drawing in image coordinates
            # multi_hand_world_landmarks is used for the real-world 3D data

            # Use multi_hand_world_landmarks for the CSV data
            if results.multi_hand_world_landmarks:
                handedness = results.multi_handedness or []
                
                # --- CRITICAL CORRECTION ON LINE 87 ---
                for idx, world_landmarks in enumerate(results.multi_hand_world_landmarks):
                    coords: list[float] = []
                    for landmark in _HAND_LANDMARKS:
                        # 'lm' will contain the world coordinates (x, y, z in meters)
                        lm = world_landmarks.landmark[landmark]
                        coords.extend([lm.x, lm.y, lm.z])

                    label = None
                    score = None
                    if idx < len(handedness) and handedness[idx].classification:
                        classification = handedness[idx].classification[0]
                        label = classification.label.lower()
                        score = classification.score

                    rows.append([time_sec, frame_index, idx, label, score] + coords)

                    # --- Use multi_hand_landmarks for drawing (image coordinates) ---
                    if writer is not None and results.multi_hand_landmarks:
                        drawing.draw_landmarks(
                            frame,
                            results.multi_hand_landmarks[idx], # Get the corresponding image landmarks
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
        df = pd.DataFrame(rows, columns=build_columns())
        df.to_csv(output_csv, index=False)
        print(f"Processed {frame_index} frames; stored {len(rows)} hand world coordinate detections.")
        print(f"World landmarks written to {output_csv.resolve()}")
    else:
        df = pd.DataFrame(columns=build_columns())
        df.to_csv(output_csv, index=False)
        print("No hand landmarks detected; wrote header-only CSV.")
        print(f"Path: {output_csv.resolve()}")

    if write_video:
        print(f"Preview video written to {out_path.resolve()}")


if __name__ == "__main__":
    main()