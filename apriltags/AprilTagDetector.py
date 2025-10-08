from pupil_apriltags import Detector
import cv2
import numpy as np
from pyk4a import PyK4APlayback, CalibrationType
import os

class AprilTagDetector:
    TAG_IDS = {0, 1}
    TAG_SIZE_M = 0.100
    MAGIC_WEIGHT = 0.65

    BOX_HALF_X = 0.190 * MAGIC_WEIGHT
    BOX_HALF_Y = 0.270 * MAGIC_WEIGHT
    BOX_HALF_Z = 0.064 * MAGIC_WEIGHT

    MIDPOINT_OFFSETS = {
        0: (0.0, 0.0, 0.048),
        1: (0.0, 0.0, 0.048),
    }

    def __init__(self, debug=True):
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
        self.playback = None
        self.writer = None

        hx, hy, hz = self.BOX_HALF_X, self.BOX_HALF_Y, self.BOX_HALF_Z
        self._verts_local = np.array([
            [-hx, -hy, -hz], [ hx, -hy, -hz], [ hx,  hy, -hz], [-hx,  hy, -hz],
            [-hx, -hy,  hz], [ hx, -hy,  hz], [ hx,  hy,  hz], [-hx,  hy,  hz],
        ], dtype=np.float32)

        self._edges = [(0,1),(1,2),(2,3),(3,0),
                       (4,5),(5,6),(6,7),(7,4),
                       (0,4),(1,5),(2,6),(3,7)]

    def load_camera_calibration(self, mkv_path: str):
        pb = PyK4APlayback(mkv_path)
        pb.open()
        calib = pb.calibration
        K = calib.get_camera_matrix(CalibrationType.COLOR).astype(np.float32)
        dist = calib.get_distortion_coefficients(CalibrationType.COLOR).astype(np.float32)

        self.K, self.dist = K, dist
        self.playback = pb

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
            out.append((tid, R, t, d))
        return out

    def _draw_box(self, img_bgr, R, t, tag_id):

        offset = np.array(self.MIDPOINT_OFFSETS[tag_id], np.float32).reshape(3,1)
        box_mid_cam = t + R @ offset

        verts_cam = (self._verts_local @ R.T) + box_mid_cam.ravel()[None,:]

        zeros3 = np.zeros((3,1), np.float32)
        mid2d, _ = cv2.projectPoints(box_mid_cam[None,:,:], zeros3, zeros3, self.K, self.dist)
        pts2d, _ = cv2.projectPoints(verts_cam.astype(np.float32), zeros3, zeros3, self.K, self.dist)

        pm = tuple(np.round(mid2d[0,0]).astype(int))
        cv2.circle(img_bgr, pm, 6, (0,255,255), -1)

        for i in range(8):
            p = tuple(np.round(pts2d[i,0]).astype(int))
            cv2.circle(img_bgr, p, 3, (0,255,0), -1)
        for a,b in self._edges:
            pa = tuple(np.round(pts2d[a,0]).astype(int))
            pb = tuple(np.round(pts2d[b,0]).astype(int))
            cv2.line(img_bgr, pa, pb, (0,255,0), 1)

    def get_poses_from_image(self, img_bgr, save_video):
        if self.K is None:
            return []

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        results = self._detect_and_estimate(gray)

        for tid, R, t, det in results:
            corners = det.corners.astype(int)
            cv2.polylines(img_bgr, [corners], True, (0,255,0), 1)
            c = tuple(np.round(det.center).astype(int))
            cv2.circle(img_bgr, c, 3, (0,255,0), -1)
            self._draw_box(img_bgr, R, t, tid)

        if self.debug:
            cv2.imshow("AprilTags", img_bgr)
            cv2.waitKey(1)

        return [{"tag_id": tid, "R": R, "t": t} for (tid, R, t, _) in results]

    def run_on_mkv(self, path: str, save_video: bool):
        if self.K is None:
            self.load_camera_calibration(path)

        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise RuntimeError("Failed to open file")
        
        out_path = None
        if save_video:
            root, _ = os.path.splitext(path)
            out_path = root + "_viz.mp4"

        initialized_writer = False

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            self.get_poses_from_image(frame_bgr, save_video)

            if save_video:
                if not initialized_writer:
                    h, w = frame_bgr.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self.writer = cv2.VideoWriter(out_path, fourcc, 30, (w, h))
                    if not self.writer.isOpened():
                        cap.release()
                        raise RuntimeError("Failed to open video writer for output")
                    initialized_writer = True

                self.writer.write(frame_bgr)

        cap.release()
        if self.writer is not None:
            print(f"Wrote visualization file to {out_path}")
            self.writer.release()
            self.writer = None
        cv2.destroyAllWindows()
