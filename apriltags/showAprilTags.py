from AprilTagDetector import AprilTagDetector
import os
from pathlib import Path

def main():
    detector = AprilTagDetector()

    folder = "trials/3"
    filename = os.listdir(folder)[5]
    path = os.path.join(folder, filename)
    if os.path.isfile(path):
        print("Target File:", path)
    base_dir = Path(__file__).parent.parent  
    full_path = os.path.join(base_dir, path)

    detector.run_on_mkv(full_path)

if __name__ == "__main__":
    main()