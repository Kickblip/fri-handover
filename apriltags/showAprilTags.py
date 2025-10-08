from AprilTagDetector import AprilTagDetector
import os
from pathlib import Path
import argparse

def main():
    detector = AprilTagDetector()

    parser = argparse.ArgumentParser(prog='AprilTagDetector',)

    parser.add_argument('filename')
    args = parser.parse_args()
    path = args.filename
    
    if os.path.isfile(path):
        print("Target File:", path)
    base_dir = Path(__file__).parent.parent  
    full_path = os.path.join(base_dir, path)

    detector.run_on_mkv(full_path)

if __name__ == "__main__":
    main()