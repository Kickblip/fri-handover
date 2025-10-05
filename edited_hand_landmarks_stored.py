
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import pandas as pd


_HAND_LANDMARKS = list(mp.solutions.hands.HandLandmark)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe Hand landmarks (21 joints) from a video and store them in a CSV."
    )
    parser.add_argument(
        "--input-video",
        required=True,
        type=Path,
        help="Path to the RGB video to process (e.g. Azure Kinect color stream).",
    )
    parser.add_argument(
        "--output-csv",
        default=Path("hand_landmarks.csv"),
        type=Path,
        help="Destination CSV file for landmark data (default: hand_landmarks.csv).",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        help="Optional debug video (mp4) with landmarks drawn on each frame.",
    )
    parser.add_argument(
        "--max-hands",
        type=int,
        default=2,
        help="Maximum number of hands to track (default: 2).",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="Minimum detection confidence threshold for hand presence.",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.5,
        help="Minimum tracking confidence to continue hand tracking.",
    )
    return parser.parse_args()


def build_columns() -> list[str]:
    columns = ["time_sec", "frame_index", "hand_index", "hand_label", "hand_score"]
    for landmark in _HAND_LANDMARKS:
        name = landmark.name.lower()
        columns.extend([f"{name}_x", f"{name}_y", f"{name}_z"])
    return columns


def main() -> None:
    args = parse_args()
    if not args.input_video.exists():
        raise FileNotFoundError(f"Input video not found: {args.input_video}")

    cap = cv2.VideoCapture(str(args.input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video {args.input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    write_video = args.output_video is not None
    if write_video:
        out_path = args.output_video.with_suffix(".mp4")
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
        max_num_hands=args.max_hands,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
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

            if results.multi_hand_landmarks:
                handedness = results.multi_handedness or []
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    coords: list[float] = []
                    for landmark in _HAND_LANDMARKS:
                        lm = hand_landmarks.landmark[landmark]
                        coords.extend([lm.x, lm.y, lm.z])

                    label = None
                    score = None
                    if idx < len(handedness) and handedness[idx].classification:
                        classification = handedness[idx].classification[0]
                        label = classification.label.lower()
                        score = classification.score

                    rows.append(
                        [time_sec, frame_index, idx, label, score] + coords
                    )

                if writer is not None:
                    for hand_landmarks in results.multi_hand_landmarks:
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
            elif writer is not None:
                # No hands; still write raw frame for timeline continuity.
                pass

            if writer is not None:
                writer.write(frame)
    finally:
        cap.release()
        hands.close()
        if writer is not None:
            writer.release()

    if rows:
        df = pd.DataFrame(rows, columns=build_columns())
        df.to_csv(args.output_csv, index=False)
        print(f"Processed {frame_index} frames; stored {len(rows)} hand detections.")
        print(f"Landmarks written to {args.output_csv.resolve()}")
    else:
        df = pd.DataFrame(columns=build_columns())
        df.to_csv(args.output_csv, index=False)
        print("No hand landmarks detected; wrote header-only CSV.")
        print(f"Path: {args.output_csv.resolve()}")

    if write_video:
        print(f"Preview video written to {out_path.resolve()}")


if __name__ == "__main__":
    main()
