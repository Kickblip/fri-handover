from pupil_apriltags import Detector
import cv2
import numpy as np
from pyk4a import PyK4APlayback, CalibrationType

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
        self.camera_calibration = {}

    def load_camera_calibration(self, mkv_path: str):
        pb = PyK4APlayback(mkv_path)
        pb.open()
        calib = pb.calibration

        # instrinsics
        K_color = calib.get_camera_matrix(CalibrationType.COLOR) # shape (3,3)
        K_depth = calib.get_camera_matrix(CalibrationType.DEPTH) # shape (3,3)

        # distortion coefficients
        dist_color = calib.get_distortion_coefficients(CalibrationType.COLOR)
        dist_depth = calib.get_distortion_coefficients(CalibrationType.DEPTH)

        # extrinsics: depth -> color (rotation R, translation t in meters)
        R_dc, t_dc = calib.get_extrinsic_parameters(CalibrationType.DEPTH, CalibrationType.COLOR) # R:(3,3), t:(3,)

        # convenience: fx, fy, cx, cy
        fx_c, fy_c, cx_c, cy_c = K_color[0,0], K_color[1,1], K_color[0,2], K_color[1,2]
        fx_d, fy_d, cx_d, cy_d = K_depth[0,0], K_depth[1,1], K_depth[0,2], K_depth[1,2]

        calibration = {
            "color": {
                "K": K_color, "dist": dist_color,
                "fx": fx_c, "fy": fy_c, "cx": cx_c, "cy": cy_c
            },
            "depth": {
                "K": K_depth, "dist": dist_depth,
                "fx": fx_d, "fy": fy_d, "cx": cx_d, "cy": cy_d
            },
            "extrinsics_depth_to_color": {"R": R_dc, "t": t_dc},
            "raw_calibration": calib,
            "playback": pb,
        }

        print("Color K:\n", calibration["color"]["K"])
        print("Color dist:", calibration["color"]["dist"])
        print("Depth->Color R:\n", calibration["extrinsics_depth_to_color"]["R"])
        print("Depth->Color t (m):", calibration["extrinsics_depth_to_color"]["t"])

        self.camera_calibration = calibration.copy()
        return calibration

    def get_poses_from_image(self, img):
        if not bool(self.camera_calibration):
            return
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        detections = self.detector.detect(gray)
        for tag in detections:
            corners = tag.corners.astype(int)
            cv2.polylines(
                img,
                [corners],
                color=(0, 0, 255),
                isClosed=True,
                thickness=2
            )
            
            c = tuple(np.round(tag.center).astype(int))
            cv2.circle(img, center=c, radius=4, color=(0, 0, 255), thickness=-1)
        
        cv2.imshow("AprilTags", img)
        cv2.waitKey(1)
        return detections