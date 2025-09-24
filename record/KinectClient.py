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


    def start_mediapipe_recording(self, path: str, n_seconds: int, hands) -> None:
        mp_drawing = mp.solutions.drawing_utils
        mp_styles = mp.solutions.drawing_styles

        record = PyK4ARecord(device=self.device, config=self.config, path=path)
        record.create()

        print(f"Recording for {n_seconds} seconds...")

     
        for i in range(30 * n_seconds):
            try:
                # Use a timeout for capture to avoid indefinite hangs on bad frames.
                # A timeout of 500 milliseconds (half a second) is a reasonable starting point.
                capture = self.device.get_capture(timeout=500)
            except K4ATimeoutException as e:
                print(f"Frame {i}: Timeout exception during capture: {e}. Skipping frame.")
                continue
            
            # --- Comprehensive Frame Validation ---
            frame = capture.color

            # Check if frame is a valid numpy array with correct properties.
            if not isinstance(frame, np.ndarray) or frame is None or frame.size == 0 or frame.ndim < 2:
                print(f"Frame {i}: Skipping invalid, empty, or malformed frame.")
                continue
            
            # Explicitly check for valid dimensions (width and height).
            # The 'remap' error is often caused by an image with zero dimensions.
            if frame.shape[0] == 0 or frame.shape[1] == 0:
                print(f"Frame {i}: Skipping invalid or empty frame with zero dimensions.")
                continue

            # --- Processing the valid frame ---
            try:
                # Convert BGRA (likely output from MJPG) to RGB (MediaPipe expects RGB).
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            except cv2.error as e:
                print(f"Frame {i}: cv2 conversion error: {e}. Skipping frame.")
                continue


            image_rgb.flags.writeable = False
            results = hands.process(image_rgb)
            

            # Convert back to BGR for display
            image_rgb.flags.writeable = True
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image_bgr,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

            # Show live preview
            cv2.imshow("Kinect + MediaPipe Hands", cv2.flip(image_bgr, 1))

            # # Convert the annotated image_bgr (NumPy array) back to a k4a.Image object
            # annotated_color_image = k4a.Image.create_from_buffer(
            #     k4a.ImageFormat.COLOR_BGRA32,
            #     image_bgr.shape[1],
            #     image_bgr.shape[0],
            #     image_bgr.shape[1] * 4, # Stride: width * number of channels (BGRA = 4)
            #     image_bgr.tobytes()
            # )

            # Overwrite the color frame in the capture object with the annotated k4a.Image
            #capture.color = annotated_color_image

            # Overwrite color frame in capture with annotated one (optional)
            #capture.color = frame  # or image_bgr if you want to save annotated frame
            if cv2.waitKey(5) & 0xFF == 27:
                break
            # Save raw capture (still includes depth, etc.)
            record.write_capture(capture)

        record.flush()
        record.close()
        print(f"{record.captures_count} frames written.")

            
    def close(self):
        if self.device:
            self.device.stop()
            self.device.close()

