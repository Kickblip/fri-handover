import os
import argparse
from PipelineManager import PipelineManager
import glob

def resolve_input_files(path):
    if os.path.isfile(path):
        return [path]

    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.mkv")))
        if not files:
            raise FileNotFoundError("No .mkv files found in directory.")
        return files

    raise FileNotFoundError(f"Invalid path: {path}")

def main():
  
    parser = argparse.ArgumentParser(
        prog="Pipeline",
        description="Run AprilTag / Hand detection on a file or directory of MKV files."
    )

    parser.add_argument(
        "path",
        help="Input file or directory. If a directory is provided, all .mkv files will be processed."
    )

    parser.add_argument(
        "-p", "--preview",
        action="store_true",
        help="Enable preview mode (show annotations while processing)."
    )
 
    args = parser.parse_args()
    path = args.path
    preview = args.preview

    pipeline = PipelineManager()

    if os.path.isdir(path):
        mkv_files = [
            os.path.join(path, f)
            for f in sorted(os.listdir(path))
            if f.lower().endswith(".mkv")
        ]

        if not mkv_files:
            raise SystemExit(f"No .mkv files found in directory: {path}")

        for mkv_path in mkv_files:
            pipeline.run_on(mkv_path, preview)

    else:
        if not os.path.isfile(path):
            raise SystemExit(f"Path does not exist: {path}")

        pipeline.run_on(path, preview)

if __name__ == "__main__":
    main()