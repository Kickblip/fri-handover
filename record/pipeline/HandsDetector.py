import cv2
from pyk4a import ImageFormat
import mediapipe as mp
import os
import csv
import numpy as np
from pyk4a import CalibrationType
from typing import List
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

class HandsDetector:

    def __init__(self, path, playback, debug=True):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        base_options = mp_tasks.BaseOptions(model_asset_path="./record/pipeline/hand_landmarker.task")
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.VIDEO,
        )
        self.hands = vision.HandLandmarker.create_from_options(options)

        self.fps = 30.0
        self.calib = playback.calibration

        self.csv_file = None
        self.csv_writer = None
        self._open_csv(path)
    
    def to_3d(self, calib, u: int, v: int, depth_mm: float) -> np.ndarray:
        """
        Convert pixel coordinate + depth to 3D using Azure Kinect calibration.
        Returns (x,y,z) in meters. If invalid, returns NaNs.
        """
        if depth_mm <= 0:
            return np.array([np.nan, np.nan, np.nan], dtype=np.float32)

        try:
            x_mm, y_mm, z_mm = calib.convert_2d_to_3d(
                (u, v),
                float(depth_mm),
                CalibrationType.COLOR,
                CalibrationType.COLOR,
            )
            return np.array([x_mm, y_mm, z_mm], dtype=np.float32) / 1000.0  # mm to m
        except Exception:
            return np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    
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
        for h in range(2): # h0, h1
            for i in range(21): # 21 landmarks per hand
                header += [
                    f"h{h}_lm{i}_x",
                    f"h{h}_lm{i}_y",
                    f"h{h}_lm{i}_z",
                ]
        self.csv_writer.writerow(header)

    def _close_csv(self):
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None

    def run_on_frame(self, capture, color_format, frame_idx, visualization_frame):
        depth = capture.transformed_depth
        if depth is None:
            depth = capture.depth
            if depth is None:
                self.frames_xyz.append([])
                return
            
        color = self.convert_to_bgra_if_required(color_format, capture.color)
        frame_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(frame_idx * 1000.0 / self.fps)
        results = self.hands.detect_for_video(mp_image, timestamp_ms)

        hands_xyz: List[np.ndarray] = []
        h, w, _ = color.shape

        if results.hand_landmarks:
            for lm_set in results.hand_landmarks:
                lm_list = landmark_pb2.NormalizedLandmarkList()
                lm_list.landmark.extend(
                    [
                        landmark_pb2.NormalizedLandmark(
                            x=lm.x,
                            y=lm.y,
                            z=lm.z,
                        )
                        for lm in lm_set
                    ]
                )
                self.mp_drawing.draw_landmarks(
                    visualization_frame,
                    lm_list,
                    self.mp_hands.HAND_CONNECTIONS,
                )
            
            for lm_set in results.hand_landmarks:
                pts = []

                for lm in lm_set:
                    u = int(lm.x * w)
                    v = int(lm.y * h)

                    if 0 <= u < w and 0 <= v < h:
                        d_mm = float(depth[v, u])
                        xyz = self.to_3d(self.calib, u, v, d_mm)
                    else:
                        xyz = np.array([np.nan, np.nan, np.nan], dtype=np.float32)

                    pts.append(xyz)

                pts_arr = np.array(pts, dtype=np.float32)
                hands_xyz.append(pts_arr)

        else:
            hands_xyz = []

        self._write_frame_row(frame_idx, hands_xyz)

        return visualization_frame
    
    def _write_frame_row(self, frame_idx: int, hands_xyz: List[np.ndarray]) -> None:
        if self.csv_writer is None:
            return
        
        row = [frame_idx]

        for h in range(2):
            if h < len(hands_xyz):
                pts = hands_xyz[h]
            else:
                pts = np.full((21, 3), np.nan, dtype=np.float32)

            for i in range(21):
                x, y, z = pts[i]
                row.extend([float(x), float(y), float(z)])
        
        self.csv_writer.writerow(row)

    def clear(self):

        self._close_csv()
        if self.hands is not None:
            self.hands.close()
            self.hands = None

      