from AprilTagDetector import AprilTagDetector
import cv2 
import os
from pathlib import Path

def process_and_show(path: str, detector: AprilTagDetector) -> None:
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError("Failed to open file")

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        detector.get_poses_from_image(frame_bgr)

    cap.release()
    cv2.destroyAllWindows()

def main():
    detector = AprilTagDetector()

    folder = "trials/2"
    filename = os.listdir(folder)[0]
    path = os.path.join(folder, filename)
    if os.path.isfile(path):
        print("Target File:", path)
    base_dir = Path(__file__).parent.parent  
    full_path = os.path.join(base_dir, path)

    calibration = detector.load_camera_calibration(full_path)
    process_and_show(full_path, detector)

if __name__ == "__main__":
    main()