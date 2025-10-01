from pupil_apriltags import Detector
import cv2
import numpy as np
from pyk4a import PyK4APlayback, CalibrationType
import math

def rvec_to_euler_xyz(rvec):
    R, _ = cv2.Rodrigues(rvec)
    
    sy = math.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0.0
    return np.degrees([x, y, z])

class AprilTagDetector():
    def __init__(self):
        self.detector = Detector(
            families="tag25h9",
            nthreads=4,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0
        )
        self.tag_sizes = {
            0: 0.100, 
            1: 0.100,  
            2: 0.064, 
            3: 0.064,
            4: 0.064,
            5: 0.064
        }
        # position of every tag relative to the center of the box
        # box laying flat on a table with 0 pointing upwards
        # 3 --> 2 = X-axis
        # 5 --> 4 = Y-axis
        # up/down is Z-axis
        self.midpoint_transforms = {
            0: (0.0, 0.0, 0.032),
            1: (0.0, 0.0, -0.032), 
            2: (-0.095, 0.0, 0.0),
            3: (0.095, 0.0, 0.0),
            4: (-0.093, 0.135, 0.0),
            5: (0.093, -0.135, 0.0)
        }
        self.camera_calibration = {}
        self.playback = None
    
    def load_camera_calibration(self, mkv_path: str):
        pb = PyK4APlayback(mkv_path)
        pb.open()
        calib = pb.calibration

        K_color = calib.get_camera_matrix(CalibrationType.COLOR).astype(np.float32)
        dist_color = calib.get_distortion_coefficients(CalibrationType.COLOR).astype(np.float32)

        fx_c, fy_c, cx_c, cy_c = K_color[0,0], K_color[1,1], K_color[0,2], K_color[1,2]

        calibration = {
            "color": {"K": K_color, "dist": dist_color, "fx": fx_c, "fy": fy_c, "cx": cx_c, "cy": cy_c},
            "raw_calibration": calib,
        }

        # print("Color K:\n", K_color)
        # print("Color dist:", dist_color)

        self.camera_calibration = calibration
        self.playback = pb
        return calibration
    
    def _pnp_pose_for_tag(self, tag, tag_size, K, dist):
        half = tag_size / 2.0
        objp = np.array([
            [-half,  half, 0.0], # top-left
            [ half,  half, 0.0], # top-right
            [ half, -half, 0.0], # bottom-right
            [-half, -half, 0.0], # bottom-left
        ], dtype=np.float32)

        imgp = tag.corners.astype(np.float32)

        success, rvec, tvec = cv2.solvePnP(
            objp, imgp, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if not success:
            return None, None
        return rvec, tvec

    def get_poses_from_image(self, img):
        if not self.camera_calibration:
            return []
        
        K = self.camera_calibration["color"]["K"]
        dist = self.camera_calibration["color"]["dist"]
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        detections = self.detector.detect(gray)
        # for tag in detections:
        #     corners = tag.corners.astype(int)
        #     cv2.polylines(
        #         img,
        #         [corners],
        #         color=(0, 0, 255),
        #         isClosed=True,
        #         thickness=2
        #     )
            
        #     c = tuple(np.round(tag.center).astype(int))
        #     cv2.circle(img, center=c, radius=4, color=(0, 0, 255), thickness=-1)
        
        box_pts = []
        results = []
        for tag in detections:
            tag_id = int(tag.tag_id)
            tag_size = self.tag_sizes.get(tag_id, None)
            if tag_size is None:
                pose_info = None
            else:
                rvec, tvec = self._pnp_pose_for_tag(tag, tag_size, K, dist)
                if rvec is not None:
                    if tag_id in self.midpoint_transforms:
                        R, _ = cv2.Rodrigues(rvec)
                        offset = np.array(self.midpoint_transforms[tag_id], dtype=np.float32).reshape(3,1)  # tag->box
                        box_cam = tvec.reshape(3,1) + R @ offset  # camera frame
                        box_pts.append(box_cam.ravel())

                    euler = rvec_to_euler_xyz(rvec)
                    pose_info = {
                        "id": tag_id,
                        "t": tvec.reshape(-1),
                        "rvec": rvec.reshape(-1),
                        "euler_xyz_deg": euler,
                        "size_m": tag_size,
                        "center_px": tag.center.tolist(),
                    }
                    results.append(pose_info)
                else:
                    pose_info = None

        box_pts, ref_R = [], None

        # for tag in detections:
        if len(detections) != 0:
            tag = detections[0]
            tag_id = int(tag.tag_id)
            tag_size = self.tag_sizes.get(tag_id, None)
            if tag_size is not None:
                rvec, tvec = self._pnp_pose_for_tag(tag, tag_size, K, dist)

                if rvec is not None:
                    R, _ = cv2.Rodrigues(rvec)
                    if tag_id in self.midpoint_transforms:
                        offset = np.array(self.midpoint_transforms[tag_id], dtype=np.float32).reshape(3,1)  # tagbox (m)
                        box_cam = tvec.reshape(3,1) + R @ offset  # box midpoint in camera frame
                        box_pts.append(box_cam.ravel())
                        if ref_R is None:
                            ref_R = R

        if box_pts and ref_R is not None:
            box_mid = np.mean(np.vstack(box_pts), axis=0).astype(np.float32).reshape(1,3)

            # half sizes (m)
            hx, hy, hz = 0.190/2.0, 0.270/2.0, 0.064/2.0

            verts_local = np.array([
                [-hx, -hy, -hz], [ hx, -hy, -hz], [ hx,  hy, -hz], [-hx,  hy, -hz],
                [-hx, -hy,  hz], [ hx, -hy,  hz], [ hx,  hy,  hz], [-hx,  hy,  hz],
            ], dtype=np.float32)

            verts_cam = (verts_local @ ref_R.T) + box_mid
            # verts_cam = (verts_local @ ref_R.T)

            # project midpoint and vertices
            zeros3 = np.zeros((3,1), np.float32)
            mid2d, _ = cv2.projectPoints(box_mid.reshape(1,1,3), zeros3, zeros3, K, dist)
            # mid2d, _ = cv2.projectPoints(box_pts[0].reshape(1,1,3), zeros3, zeros3, K, dist)
            pts2d, _ = cv2.projectPoints(verts_cam.astype(np.float32), zeros3, zeros3, K, dist)

            # draw midpoint
            pm = tuple(np.round(mid2d[0,0]).astype(int))
            cv2.circle(img, pm, 6, (0, 255, 255), -1)

            # draw vertices
            for i in range(8):
                p = tuple(np.round(pts2d[i,0]).astype(int))
                cv2.circle(img, p, 4, (0, 0, 255), -1)

            # draw edges
            edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
            for a, b in edges:
                pa = tuple(np.round(pts2d[a,0]).astype(int))
                pb = tuple(np.round(pts2d[b,0]).astype(int))
                cv2.line(img, pa, pb, (0, 0, 255), 1)
        
        cv2.imshow("AprilTags", img)
        cv2.waitKey(1)
        return results
    
    def run_on_mkv(self, path: str) -> None:
        if not self.camera_calibration:
            self.load_camera_calibration(path)

        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise RuntimeError("Failed to open file")

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            self.get_poses_from_image(frame_bgr)

        cap.release()
        cv2.destroyAllWindows()
