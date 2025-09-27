from AprilTagDetector import AprilTagDetector
import numpy as np
import cv2 
import os
from pathlib import Path

def process_and_show(video_path, detector):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = 1.0 / fps if fps > 0 else 1/30

    frames = []

    # Pass 1: process & store
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = detector.getPoses(gray)

        for tag, (R, t) in detections:
            c_x, c_y = map(int, tag.center)
            cv2.circle(frame, (c_x, c_y), 6, (0, 0, 255), -1)
            cv2.putText(frame, f"ID:{tag.tag_id}", (c_x+10, c_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        frames.append(frame)

    cap.release()

    # Pass 2: show quickly
    for frame in frames:
        cv2.imshow("Processed Video", frame)
        if cv2.waitKey(int(delay * 1000)) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def main():
    detector = AprilTagDetector()

    folder = "trials/2"
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if os.path.isfile(path):
            print(path)
        base_dir = Path(__file__).parent.parent  
        full_path = os.path.join(base_dir, path)
        process_and_show(full_path, detector)

if __name__ == "__main__":
    main()