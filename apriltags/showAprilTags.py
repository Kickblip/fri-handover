from AprilTagDetector import AprilTagDetector
import numpy as np
import cv2 
import os
from pathlib import Path


def main():
    detector = AprilTagDetector()

    folder = "trials/2"
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if os.path.isfile(path):
            print(path)
        base_dir = Path(__file__).parent.parent  
        full_path = os.path.join(base_dir, path)
        print(full_path)

        cap = cv2.VideoCapture(full_path)

        if not cap.isOpened():
            print(f"Error: Could not open {path}")
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break 

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3,3), 0)


            detections = detector.getPoses(gray) 

            for tag, pose in detections:
                c_x, c_y = map(int, tag.center)
                cv2.circle(frame, (c_x, c_y), 6, (0, 0, 255), -1)
                cv2.putText(frame, f"ID:{tag.tag_id}", (c_x+10, c_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("AprilTag Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()