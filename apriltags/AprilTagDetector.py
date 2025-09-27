from pupil_apriltags import Detector
import numpy as np
import math
class AprilTagDetector():
    def __init__(self):
        self.detector =  Detector(
        families="tag25h9",   # change if your tags are another family
        nthreads=4,
        quad_decimate=1.0,     # keep resolution (good for small tags)
        quad_sigma=0.0,
        refine_edges=1
    )
        self.K_color = np.array([
            [600.0,   0.0, 320.0],
            [  0.0, 600.0, 240.0],
            [  0.0,   0.0,   1.0]
        ])
        self.tag_sizes = {
            0: 0.100, 
            1: 0.100,  
            2: 0.064, 
            3: 0.064,
            4: 0.064,
            5: 0.064
        }
        
    def getPoses(self, img):
        h, w = img.shape[:2]
        cx, cy = w/2.0, h/2.0
        fx = fy = w / (2 * math.tan(math.radians(60) / 2))
        result = []
        for tag_id, tag_size in self.tag_sizes.items():
            detections = self.detector.detect(
                img,
                estimate_tag_pose=True,
                camera_params=[fx, fy, cx, cy],
                tag_size=tag_size
            )
            for tag in detections:
                result.append((tag, (tag.pose_R, tag.pose_t)))
        return result


