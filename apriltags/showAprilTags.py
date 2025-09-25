import AprilTagDetector
import numpy as np
import cv2 


def main():
    detector = AprilTagDetector()
    video_path = "/Users/diego/Programming/fri-handover/trials/1/1_diego_rohan.mkv"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break 

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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