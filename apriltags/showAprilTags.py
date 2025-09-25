import AprilTagDetector
import numpy as np
import cv2


def main():
    detector = AprilTagDetector()

    while True:
        capture = detector.device.get_capture()
        if capture.color is None:
            continue

        # Convert color to OpenCV BGR
        img_bgr = capture.color[:, :, :3]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Detect tags and poses
        detections = detector.getPoses(gray)

        # Draw red dot at each tag center
        for tag, pose in detections:
            c_x, c_y = int(tag.center[0]), int(tag.center[1])
            cv2.circle(img_bgr, (c_x, c_y), 6, (0, 0, 255), -1)  # red dot
            cv2.putText(img_bgr, f"ID:{tag.tag_id}", (c_x+10, c_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("AprilTag Detection", img_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    detector.device.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()