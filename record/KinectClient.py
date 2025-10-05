from matplotlib.pylab import record
from pyk4a import Config, ImageFormat, PyK4A, PyK4ARecord, ColorResolution, DepthMode, FPS
import cv2
import mediapipe as mp
import time
import numpy as np

# import k4a

class KinectClient:
    def __init__(self):
        self.config = Config(
            # https://unanancyowen.github.io/k4asdk_python_apireference/classk4a_1_1__bindings_1_1k4atypes_1_1_device_configuration.html
            color_format=ImageFormat.COLOR_BGRA32,
            color_resolution=ColorResolution.RES_1080P,
            depth_mode=DepthMode.NFOV_UNBINNED,
            camera_fps=FPS.FPS_30,
            synchronized_images_only=True,
        )
        self.device = PyK4A(config=self.config, device_id=0)
        self.device.start()

    def start_recording(self, path: str, n_seconds: int = 5) -> None:
        record = PyK4ARecord(device=self.device, config=self.config, path=path)
        record.create()

        print(f"Recording for {n_seconds} seconds...")
        for _ in range(30 * n_seconds):
            capture = self.device.get_capture()
            record.write_capture(capture)

        record.flush()
        record.close()
        print(f"{record.captures_count} frames written.")

            
    def close(self):
        if self.device:
            self.device.stop()
            self.device.close()

