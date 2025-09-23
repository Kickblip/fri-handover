from pyk4a import CalibrationType, 
import pupil_apriltags as apriltag
class AprilTagDetector():
    __init__():
        self.detector = apriltag.Detector()
        self.device = PyK4A(config=self.config, device_id=0)
        calibration = device.calibration
        self.K_color = calibration.get_camera_matrix(CalibrationType.COLOR)
        self.tag_sizes = {
            0: 0.100, 
            1: 0.100,  
            2: 0.064, 
            3: 0.064,
            4: 0.064,
            5: 0.064
        }
        
    def getPoses(img):

        tags = self.detector.detect(img)
        fx, fy = self.K_color[0, 0], self.K_color[1, 1]
        cx, cy = self.K_color[0, 2], self.K_color[1, 2]

        result = []
        for tag in tags:
            if tag.tag_id not in self.tag_sizes:
                print(f"Warning: No size found for tag {tag.tag_id}, skipping")
                continue

            tag_size = tag_sizes[tag.tag_id]

            pose, e0, e1 = detector.detection_pose(
                tag, (fx, fy, cx, cy), tag_size
            )
            result.append((tag.tag_id, pose))

        return result


