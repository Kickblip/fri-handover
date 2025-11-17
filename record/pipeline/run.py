import os
from pathlib import Path
import argparse
from PipelineManager import PipelineManager

def main():

    pipeline = PipelineManager()
  
    parser = argparse.ArgumentParser(prog='AprilTagDetector',)

    parser.add_argument('filename')
    args = parser.parse_args()
    path = args.filename

    if os.path.isfile(path):
        print("Target File:", path)
    base_dir = Path(__file__).parent.parent.parent 
    full_path = os.path.join(base_dir, path)

    pipeline.run_on(full_path)

    # pass one file at a time to the manager
    # give the manager input file path and output filenames

if __name__ == "__main__":
    main()