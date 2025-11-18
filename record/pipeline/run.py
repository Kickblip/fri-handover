import os
import argparse
from PipelineManager import PipelineManager
import glob
import sys
import subprocess

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

    if os.path.isdir(path):
        mkv_files = [
            os.path.join(path, f)
            for f in sorted(os.listdir(path))
            if f.lower().endswith(".mkv")
        ]

        if not mkv_files:
            raise SystemExit(f"No .mkv files found in directory: {path}")

        for mkv_path in mkv_files:
            cmd = [sys.executable, os.path.abspath(__file__)]
            if preview:
                cmd.append("--preview")
            cmd.append(mkv_path)

            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as exc:
                raise SystemExit(
                    f"Processing failed for {mkv_path} with exit code {exc.returncode}"
                ) from exc

    else:
        if not os.path.isfile(path):
            raise SystemExit(f"Path does not exist: {path}")
        
        pipeline = PipelineManager()
        pipeline.run_on(path, preview)

if __name__ == "__main__":
    main()