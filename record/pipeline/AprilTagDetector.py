from pupil_apriltags import Detector
import cv2
import numpy as np
from pyk4a import CalibrationType
import os
import csv
from pyk4a import ImageFormat

class AprilTagDetector:
    TAG_IDS = {0, 1}
    TAG_SIZE_M = 0.100
    MAGIC_WEIGHT = 0.64

    BOX_HALF_X = 0.190 * MAGIC_WEIGHT
    BOX_HALF_Y = 0.270 * MAGIC_WEIGHT
    BOX_HALF_Z = 0.064 * MAGIC_WEIGHT

    MIDPOINT_OFFSETS = {
        0: (0.0, 0.0, 0.048),
        1: (0.0, 0.0, 0.048),
    }

    def __init__(self, path, playback, debug=True):
        self.debug = debug
        self.detector = Detector(
            families="tag25h9",
            nthreads=4,
            quad_decimate=1.5,
            quad_sigma=0.8,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0
        )
        self.K = None
        self.dist = None
        self.playback = playback

        self.csv_file = None
        self.csv_writer = None

        hx, hy, hz = self.BOX_HALF_X, self.BOX_HALF_Y, self.BOX_HALF_Z
        self._verts_local = np.array([
            [-hx, -hy, -hz], [ hx, -hy, -hz], [ hx,  hy, -hz], [-hx,  hy, -hz],
            [-hx, -hy,  hz], [ hx, -hy,  hz], [ hx,  hy,  hz], [-hx,  hy,  hz],
        ], dtype=np.float32)

        self._edges = [(0,1),(1,2),(2,3),(3,0),
                       (4,5),(5,6),(6,7),(7,4),
                       (0,4),(1,5),(2,6),(3,7)]
        
        self.load_camera_calibration()
        self._open_csv(path)

    def load_camera_calibration(self):
        calib = self.playback.calibration
        K = calib.get_camera_matrix(CalibrationType.COLOR).astype(np.float32)
        dist = calib.get_distortion_coefficients(CalibrationType.COLOR).astype(np.float32)

        self.K, self.dist = K, dist

    def _detect_and_estimate(self, gray):
        fx, fy, cx, cy = float(self.K[0,0]), float(self.K[1,1]), float(self.K[0,2]), float(self.K[1,2])
        dets = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=(fx, fy, cx, cy),
            tag_size=self.TAG_SIZE_M
        )

        out = []
        for d in dets:
            tid = int(d.tag_id)
            if tid not in self.TAG_IDS:
                continue
            if getattr(d, "pose_R", None) is None or getattr(d, "pose_t", None) is None:
                continue
            R = np.asarray(d.pose_R, dtype=np.float32)
            t = np.asarray(d.pose_t, dtype=np.float32).reshape(3,1)
            # R, t map tag-frame points to camera frame: p_cam = R @ p_tag + t
            out.append((tid, R, t, d))
        return out

    def _compute_box_vertices_cam(self, R: np.ndarray, t: np.ndarray, tag_id: int) -> np.ndarray:
        #return (8,3) vertices of the box in camera coordinates (meters)
        offset = np.array(self.MIDPOINT_OFFSETS[tag_id], np.float32).reshape(3,1)
        # box midpoint in camera frame
        box_mid_cam = t + R @ offset  # (3,1)
        # rotate local vertices and translate to camera frame
        verts_cam = (self._verts_local @ R.T) + box_mid_cam.ravel()[None, :]
        return verts_cam

    def _draw_box(self, img_bgr, R, t, tag_id):
        offset = np.array(self.MIDPOINT_OFFSETS[tag_id], np.float32).reshape(3,1)
        box_mid_cam = t + R @ offset
        verts_cam = (self._verts_local @ R.T) + box_mid_cam.ravel()[None,:]

        zeros3 = np.zeros((3,1), np.float32)
        mid2d, _ = cv2.projectPoints(box_mid_cam[None,:,:], zeros3, zeros3, self.K, None)
        pts2d, _ = cv2.projectPoints(verts_cam.astype(np.float32), zeros3, zeros3, self.K, None)

        pm = tuple(np.round(mid2d[0,0]).astype(int))
        cv2.circle(img_bgr, pm, 6, (0,255,255), -1)

        for i in range(8):
            p = tuple(np.round(pts2d[i,0]).astype(int))
            cv2.circle(img_bgr, p, 3, (0,255,0), -1)
        for a,b in self._edges:
            pa = tuple(np.round(pts2d[a,0]).astype(int))
            pb = tuple(np.round(pts2d[b,0]).astype(int))
            cv2.line(img_bgr, pa, pb, (0,255,0), 2)

    def get_poses_from_image(self, img_bgr, frame_idx: int):
        if self.K is None:
            return []

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        results = self._detect_and_estimate(gray)

        rows_for_csv = []

        for tid, R, t, det in results:

            corners = det.corners.astype(int)
            cv2.polylines(img_bgr, [corners], True, (0,255,0), 1)
            c = tuple(np.round(det.center).astype(int))
            cv2.circle(img_bgr, c, 3, (0,255,0), -1)
            self._draw_box(img_bgr, R, t, tid)

            verts_cam = self._compute_box_vertices_cam(R, t, tid)  # (8,3)
            flat = verts_cam.reshape(-1)  # 24 values
            rows_for_csv.append([frame_idx, tid, "camera"] + flat.tolist())

        # if no detection, still write a row for this frame with NaNs
        if not rows_for_csv:
            nan_row = [frame_idx, -1, "camera"] + [float('nan')]*24
            rows_for_csv.append(nan_row)

        if self.csv_writer is not None:
            self.csv_writer.writerows(rows_for_csv)

        return img_bgr

    def _open_csv(self, mkv_path: str):
        root, _ = os.path.splitext(mkv_path)
        csv_path = root + "_box.csv"
        self.csv_file = open(csv_path, mode="w", newline="")
        self.csv_writer = csv.writer(self.csv_file)

        header = ["frame_idx", "tag_id", "coord_frame"]
        # v0_x, v0_y, v0_z, ..., v7_x, v7_y, v7_z
        for i in range(8):
            header += [f"v{i}_x", f"v{i}_y", f"v{i}_z"]
        self.csv_writer.writerow(header)

    def convert_to_bgra_if_required(self, color_format: ImageFormat, color_image):
        if color_format == ImageFormat.COLOR_MJPG:
            color_image = cv2.imdecode(color_image, cv2.IMREAD_COLOR)
        return color_image

    def _close_csv(self):
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None

    def run_on_frame(self, frame_img, frame_idx):
        frame_bgr = self.convert_to_bgra_if_required(self.playback.configuration["color_format"], frame_img.color)
        frame_with_box = self.get_poses_from_image(frame_bgr, frame_idx)
        return frame_with_box

    def clear(self):
        self._close_csv()

        

        

