from pupil_apriltags import Detector
import cv2
import numpy as np

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
        
    def getPoses(self, img):
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