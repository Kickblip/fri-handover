from AprilTagDetector import AprilTagDetector
from HandsDetector import HandsDetector
import cv2

class PipelineManager:

    def __init__(self):
        self.AprilTag = None
        self.MediaPipe = None
        self.playback = None

    def run_on(self, path: str):

        self.AprilTag = AprilTagDetector(path, debug=True)
        self.playback = self.AprilTag.get_playback()
        self.MediaPipe = HandsDetector(path, debug=True)

        frame_idx = 0

        try:
            while True:
                try:
                    capture = self.playback.get_next_capture()
                except EOFError:
                    break 
                    
                frame_with_box = self.AprilTag.run_on_frame(capture, frame_idx)

                frame_with_hands_and_box = self.MediaPipe.run_on_frame(capture, self.playback.configuration["color_format"], frame_idx, frame_with_box)

                cv2.imshow(path, frame_with_hands_and_box)
                cv2.waitKey(1)
                
                frame_idx += 1

        finally:
            self.AprilTag.clear()
            self.MediaPipe.clear()
