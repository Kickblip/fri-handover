import cv2
from pyk4a import ImageFormat
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import os
import csv

class HandsDetector:

    def __init__(self, path, debug=True):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
        )

        self.csv_file = None
        self.csv_writer = None
        self._open_csv(path)
    
    def convert_to_bgra_if_required(self, color_format: ImageFormat, color_image):
        if color_format == ImageFormat.COLOR_MJPG:
            color_image = cv2.imdecode(color_image, cv2.IMREAD_COLOR)
        return color_image
    
    def _open_csv(self, mkv_path: str):
        root, _ = os.path.splitext(mkv_path)
        csv_path = root + "_hands.csv"
        self.csv_file = open(csv_path, mode="w", newline="")
        self.csv_writer = csv.writer(self.csv_file)

        header = ["frame_idx"]
        for i in range(21):
            header += [f"h1_{i}_x", f"h1_{i}_y"]
        for i in range(21):
            header += [f"h2_{i}_x", f"h2_{i}_y"]
        self.csv_writer.writerow(header)

    def _close_csv(self):
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None

    def run_on_frame(self, capture, color_format, frame_idx, visualization_frame):
        capture_frame = self.convert_to_bgra_if_required(color_format, capture.color)

        frame_rgb = cv2.cvtColor(capture_frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for lm_set in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    visualization_frame,
                    lm_set,
                    self.mp_hands.HAND_CONNECTIONS,
                )

        # for hand in detection_result.hand_landmarks:
        #     for landmark in hand:
        #         row_for_csv.append(landmark.x)
        #         row_for_csv.append(landmark.y)            

        # if self.csv_writer is not None:
        #         self.csv_writer.writerows(row_for_csv)

        return visualization_frame

    def clear(self):
        self._close_csv()

      