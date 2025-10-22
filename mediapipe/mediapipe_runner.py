# pipeline/run_pipeline.py
from __future__ import annotations

from pathlib import Path
import importlib
import sys
import argparse

# Make sure parent path is in sys.path for imports like "pipeline.step1_world_extract"
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Root paths
ROOT = Path(__file__).resolve().parents[1] / "dataset"
INPUT_DIR = ROOT / "input_video"

# Stage modules (replace with your module names)
STEP1 = "hand_mediapipe"          # video -> world coords + world preview video
STEP2 = "process_to_quaternion"   # world csv -> quaternions csv
STEP3 = "convert_to_rodrigues"    # quaternions csv -> rodrigues csv
STEP4 = "two_closest_points"      # world csv + video -> closest pair video + csvs


# ------------------- DIRECTORY SETUP -------------------
def ensure_dirs():
    """Create all output directories if they don't exist."""
    (ROOT / "mediapipe_outputs" / "csv" / "world").mkdir(parents=True, exist_ok=True)
    (ROOT / "mediapipe_outputs" / "csv" / "quaternions").mkdir(parents=True, exist_ok=True)
    (ROOT / "mediapipe_outputs" / "csv" / "rodrigues").mkdir(parents=True, exist_ok=True)
    (ROOT / "mediapipe_outputs" / "csv" / "all_pairs_distance").mkdir(parents=True, exist_ok=True)
    (ROOT / "mediapipe_outputs" / "csv" / "closest_pair_distance").mkdir(parents=True, exist_ok=True)
    (ROOT / "mediapipe_outputs" / "video" / "world").mkdir(parents=True, exist_ok=True)
    (ROOT / "mediapipe_outputs" / "video" / "closest_point").mkdir(parents=True, exist_ok=True)


# ------------------- MODULE LOADER -------------------
def load(module_name: str):
    """Safely import a module by name."""
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        print(f"❌ Could not import {module_name}: {e}")
        sys.exit(1)


# ------------------- MAIN PIPELINE PER VIDEO -------------------
def run_for_video(video_path: Path, step: int | None = None):
    """Run the specified (or all) pipeline stages for one video."""
    stem = video_path.stem
    print(f"\n================= {video_path.name} =================")

    # Paths derived from video filename
    world_csv    = ROOT / "mediapipe_outputs" / "csv" / "world" / f"{stem}_world.csv"
    world_vid    = ROOT / "mediapipe_outputs" / "video" / "world" / f"{stem}_world.mp4"
    quat_csv     = ROOT / "mediapipe_outputs" / "csv" / "quaternions" / f"{stem}_quaternions.csv"
    rodr_csv     = ROOT / "mediapipe_outputs" / "csv" / "rodrigues" / f"{stem}_rodrigues.csv"
    closest_vid  = ROOT / "mediapipe_outputs" / "video" / "closest_point" / f"{stem}_closest_pair.mp4"
    perframe_csv = ROOT / "mediapipe_outputs" / "csv" / "all_pairs_distance" / f"{stem}_closest_per_frame.csv"
    global_csv   = ROOT / "mediapipe_outputs" / "csv" / "closest_pair_distance" / f"{stem}_closest_global.csv"

    # ---- STEP 1: Video → World ----
    if step is None or step == 1:
        s1 = load(STEP1)
        s1.input_video = video_path
        s1.output_csv = world_csv
        s1.output_video = world_vid
        if hasattr(s1, "max_hands"): s1.max_hands = 2
        if hasattr(s1, "min_detection_confidence"): s1.min_detection_confidence = 0.5
        if hasattr(s1, "min_tracking_confidence"): s1.min_tracking_confidence = 0.5
        print("[1/4] Extracting world coordinates...")
        s1.main()

    # ---- STEP 2: World → Quaternions ----
    if step is None or step == 2:
        if not world_csv.exists():
            print(f"⚠️ Missing input: {world_csv}. Run step 1 first.")
            return
        s2 = load(STEP2)
        s2.input_csv = world_csv
        s2.output_csv = quat_csv
        print("[2/4] Converting world → quaternions...")
        s2.main()

    # ---- STEP 3: Quaternions → Rodrigues ----
    if step is None or step == 3:
        if not quat_csv.exists():
            print(f"⚠️ Missing input: {quat_csv}. Run step 2 first.")
            return
        s3 = load(STEP3)
        s3.input_quat_csv = quat_csv
        s3.output_rodrigues_csv = rodr_csv
        print("[3/4] Converting quaternions → Rodrigues...")
        s3.main()

    # ---- STEP 4: Closest pair (World CSV + Video) ----
    if step is None or step == 4:
        if not world_csv.exists():
            print(f"⚠️ Missing input: {world_csv}. Run step 1 first.")
            return
        s4 = load(STEP4)
        s4.INPUT_VIDEO_PATH = video_path
        s4.INPUT_CSV_PATH = world_csv
        s4.OUTPUT_VIDEO_PATH = closest_vid
        s4.OUTPUT_LINES_CSV_PATH = perframe_csv
        s4.OUTPUT_GLOBAL_MIN_CSV_PATH = global_csv
        print("[4/4] Computing closest pair + annotated video...")
        s4.main()

    print(f"✅ Done: {stem}")
    print(f"   world csv        -> {world_csv}")
    print(f"   world preview    -> {world_vid}")
    print(f"   quaternions csv  -> {quat_csv}")
    print(f"   rodrigues csv    -> {rodr_csv}")
    print(f"   closest per-frame-> {perframe_csv}")
    print(f"   closest global   -> {global_csv}")
    print(f"   annotated video  -> {closest_vid}")


# ------------------- MAIN ENTRY POINT -------------------
def main():
    ensure_dirs()

    parser = argparse.ArgumentParser(description="Run the FRI Handover pipeline (optionally by step or video).")
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4],
                        help="Run only this specific step (1–4). Default runs all.")
    parser.add_argument("--video", type=str,
                        help="Run only on a specific video filename (e.g., 1_video.mkv)")
    args = parser.parse_args()

    vids = sorted(INPUT_DIR.glob("*.mkv"))
    if not vids:
        print(f"⚠️ No .mkv files found in {INPUT_DIR.resolve()}")
        sys.exit(0)

    if args.video:
        vids = [v for v in vids if v.stem == Path(args.video).stem]
        if not vids:
            print(f"⚠️ Video {args.video} not found in {INPUT_DIR.resolve()}")
            sys.exit(0)

    print("FRI Handover — Full Pipeline Runner")
    print(f"Found {len(vids)} video(s): {[v.name for v in vids]}")

    for v in vids:
        run_for_video(v, step=args.step)


if __name__ == "__main__":
    main()
