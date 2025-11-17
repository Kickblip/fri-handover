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
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands

        self.csv_file = None
        self.csv_writer = None
        self._open_csv(path)
    
    def convert_to_bgra_if_required(self, color_format: ImageFormat, color_image):
        if color_format == ImageFormat.COLOR_MJPG:
            color_image = cv2.imdecode(color_image, cv2.IMREAD_COLOR)
        return color_image

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        hand_landmarks_list = detection_result.hand_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]

            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

        return annotated_image
    
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
        format_frame = cv2.cvtColor(capture_frame, cv2.COLOR_BGRA2RGBA)
        row_for_csv = [frame_idx]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=format_frame)

        base_options = python.BaseOptions(model_asset_path='./record/pipeline/hand_landmarker.task')
        options = vision.HandLandmarkerOptions(base_options=base_options,
                                            num_hands=2, running_mode=mp.tasks.vision.RunningMode.IMAGE)
        detector = vision.HandLandmarker.create_from_options(options)

        # detection_result.hand_landmarks For each detected hand, a list of 21 normalized 2D landmarks in image coordinates
        detection_result = detector.detect(mp_image)

        annotated_image = self.draw_landmarks_on_image(visualization_frame, detection_result)

        # for hand in detection_result.hand_landmarks:
        #     for landmark in hand:
        #         row_for_csv.append(landmark.x)
        #         row_for_csv.append(landmark.y)            

        # if self.csv_writer is not None:
        #         self.csv_writer.writerows(row_for_csv)

        return annotated_image

    def clear(self):
        self._close_csv()

      