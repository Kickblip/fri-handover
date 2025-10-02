from pupil_apriltags import Detector
import cv2
import numpy as np
import math
import time


def rvec_to_euler_xyz(rvec):
    R, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0.0
    return np.degrees([x, y, z])


class AprilTagDetector:
    def __init__(self, camera_matrix=None, dist_coeffs=None):
        """
        AprilTag detector that works on macOS without pyk4a.
        You must provide camera intrinsics manually if you want accurate pose estimation.
        """
        self.detector = Detector(
            families="tag25h9",
            nthreads=4,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0
        )

        # Default intrinsics (approx, 1080p webcam)
        if camera_matrix is None:
            fx = fy = 1000.0
            cx, cy = 960, 540  # image center for 1920x1080
            self.K = np.array([[fx, 0, cx],
                               [0, fy, cy],
                               [0, 0, 1]], dtype=np.float32)
        else:
            self.K = camera_matrix.astype(np.float32)

        self.dist = np.zeros((5, 1), dtype=np.float32) if dist_coeffs is None else dist_coeffs

        # Known tag sizes in meters
        self.tag_sizes = {
            0: 0.100,
            1: 0.100,
            2: 0.064,
            3: 0.064,
            4: 0.064,
            5: 0.064
        }

        # For relative timing
        self.start_time = None

    def _pnp_pose_for_tag(self, tag, tag_size):
        half = tag_size / 2.0
        objp = np.array([
            [-half,  half, 0.0],  # top-left
            [ half,  half, 0.0],  # top-right
            [ half, -half, 0.0],  # bottom-right
            [-half, -half, 0.0],  # bottom-left
        ], dtype=np.float32)

        imgp = tag.corners.astype(np.float32)

        success, rvec, tvec = cv2.solvePnP(
            objp, imgp, self.K, self.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if not success:
            return None, None
        return rvec, tvec

    def get_poses_from_image(self, img):
        if self.start_time is None:
            self.start_time = time.perf_counter()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray)

        results = []
        rel_time = time.perf_counter() - self.start_time  # relative time in seconds

        for tag in detections:
            tag_id = int(tag.tag_id)
            tag_size = self.tag_sizes.get(tag_id, None)
            if tag_size is None:
                continue

            rvec, tvec = self._pnp_pose_for_tag(tag, tag_size)
            if rvec is not None:
                euler = rvec_to_euler_xyz(rvec)
                pose_info = {
                    "id": tag_id,
                    "t": tvec.reshape(-1),
                    "rvec": rvec.reshape(-1),
                    "euler_xyz_deg": euler,
                    "center_px": tag.center.tolist(),
                    "timestamp": rel_time
                }
                results.append(pose_info)

                # --- Drawing on image ---
                corners = tag.corners.astype(int)
                cv2.polylines(img, [corners], True, (0, 0, 255), 2)
                c = tuple(np.round(tag.center).astype(int))
                cv2.circle(img, c, 4, (0, 255, 0), -1)

                # Show ID
                cv2.putText(img, f"ID {tag_id}", (c[0] + 10, c[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Show timestamp (ms precision)
                cv2.putText(img, f"{rel_time:.3f}s", (c[0] + 10, c[1] + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cv2.imshow("AprilTags", img)
        cv2.waitKey(1)
        return results

    def run_on_mkv(self, path: str) -> None:
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise RuntimeError("Failed to open MKV file")

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            self.get_poses_from_image(frame_bgr)

        cap.release()
        cv2.destroyAllWindows()
