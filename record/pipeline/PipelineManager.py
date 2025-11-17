from AprilTagDetector import AprilTagDetector
from HandsDetector import HandsDetector
import cv2
from rich.progress import Progress
import os
from pyk4a import PyK4APlayback

class PipelineManager:

    def __init__(self):
        self.AprilTag = None
        self.MediaPipe = None
        self.playback = None

    def run_on(self, path: str, preview_mode: bool):

        pb = PyK4APlayback(path)
        pb.open()
        self.playback = pb

        self.AprilTag = AprilTagDetector(path, self.playback, debug=True)
        self.MediaPipe = HandsDetector(path, self.playback, debug=True)

        frame_idx = 0

        with Progress() as progress:
            task = progress.add_task(
                f"[green]Processing {os.path.basename(path)}...", 
                total=120
            )

            try:
                while True:
                    try:
                        capture = self.playback.get_next_capture()
                    except EOFError:
                        break 
                        
                    frame_with_box = self.AprilTag.run_on_frame(capture, frame_idx)

                    frame_with_hands_and_box = self.MediaPipe.run_on_frame(capture, self.playback.configuration["color_format"], frame_idx, frame_with_box)

                    if preview_mode:
                        cv2.imshow(path, frame_with_hands_and_box)
                        cv2.waitKey(1)
                    
                    frame_idx += 1

                    progress.update(task, advance=1)

            finally:
                self.AprilTag.clear()
                self.MediaPipe.clear()
                self.playback.close()
                self.AprilTag = None
                self.MediaPipe = None
                self.playback = None

                progress.update(
                    task,
                    description=f"[green]Finished {os.path.basename(path)} ({frame_idx} frames)"
                )
