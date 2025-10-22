# pipeline/run_pipeline.py
from __future__ import annotations

from pathlib import Path
import importlib
import sys

ROOT = Path("dataset")
INPUT_DIR = ROOT / "input_video"

# stage modules (kept separate)
STEP1 = "pipeline.step1_world_extract"       # video -> world coords + world preview video
STEP2 = "pipeline.step2_world_to_quat"       # world csv -> quaternions csv
STEP3 = "pipeline.step3_quat_to_rodrigues"   # quaternions csv -> rodrigues csv
STEP4 = "pipeline.step4_closest_pair"        # world csv + video -> closest pair video + csvs

def ensure_dirs():
    (ROOT / "mediapipe_outputs" / "csv" / "world").mkdir(parents=True, exist_ok=True)
    (ROOT / "mediapipe_outputs" / "csv" / "quaternions").mkdir(parents=True, exist_ok=True)
    (ROOT / "mediapipe_outputs" / "csv" / "rodrigues").mkdir(parents=True, exist_ok=True)
    (ROOT / "mediapipe_outputs" / "csv" / "all_pairs_distance").mkdir(parents=True, exist_ok=True)
    (ROOT / "mediapipe_outputs" / "csv" / "closest_pair_distance").mkdir(parents=True, exist_ok=True)
    (ROOT / "mediapipe_outputs" / "video" / "world").mkdir(parents=True, exist_ok=True)
    (ROOT / "mediapipe_outputs" / "video" / "closest_point").mkdir(parents=True, exist_ok=True)

def load(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        print(f"❌ Could not import {module_name}: {e}")
        sys.exit(1)

def run_for_video(video_path: Path):
    stem = video_path.stem  # e.g., "1_video"
    print(f"\n================= {video_path.name} =================")

    # Paths derived from input filename:
    world_csv   = ROOT / "mediapipe_outputs" / "csv" / "world" / f"{stem}_world.csv"
    world_vid   = ROOT / "mediapipe_outputs" / "video" / "world" / f"{stem}_world.mp4"
    quat_csv    = ROOT / "mediapipe_outputs" / "csv" / "quaternions" / f"{stem}_quaternions.csv"
    rodr_csv    = ROOT / "mediapipe_outputs" / "csv" / "rodrigues" / f"{stem}_rodrigues.csv"
    closest_vid = ROOT / "mediapipe_outputs" / "video" / "closest_point" / f"{stem}_closest_pair.mp4"
    perframe_csv= ROOT / "mediapipe_outputs" / "csv" / "all_pairs_distance" / f"{stem}_closest_per_frame.csv"
    global_csv  = ROOT / "mediapipe_outputs" / "csv" / "closest_pair_distance" / f"{stem}_closest_global.csv"

    # ---- STEP 1: video -> world ----
    s1 = load(STEP1)
    # set its config, then run
    s1.input_video = video_path
    s1.output_csv = world_csv
    s1.output_video = world_vid
    if hasattr(s1, "max_hands"): s1.max_hands = 2
    if hasattr(s1, "min_detection_confidence"): s1.min_detection_confidence = 0.5
    if hasattr(s1, "min_tracking_confidence"): s1.min_tracking_confidence = 0.5
    print("[1/4] Extracting world coordinates...")
    s1.main()

    # ---- STEP 2: world -> quaternions ----
    s2 = load(STEP2)
    s2.input_csv = world_csv
    s2.output_csv = quat_csv
    if hasattr(s2, "MAX_HANDS"): s2.MAX_HANDS = 2
    print("[2/4] Converting world -> quaternions...")
    s2.main()

    # ---- STEP 3: quaternions -> Rodrigues ----
    s3 = load(STEP3)
    s3.input_quat_csv = quat_csv
    s3.output_rodrigues_csv = rodr_csv
    if hasattr(s3, "MAX_HANDS"): s3.MAX_HANDS = 2
    print("[3/4] Converting quaternions -> Rodrigues...")
    s3.main()

    # ---- STEP 4: closest-hands (uses world csv + original video) ----
    s4 = load(STEP4)
    s4.INPUT_VIDEO_PATH = video_path
    s4.INPUT_CSV_PATH = world_csv   # we compute distances in world space
    s4.OUTPUT_VIDEO_PATH = closest_vid
    s4.OUTPUT_LINES_CSV_PATH = perframe_csv
    s4.OUTPUT_GLOBAL_MIN_CSV_PATH = global_csv
    if hasattr(s4, "MAX_HANDS"): s4.MAX_HANDS = 2
    if hasattr(s4, "MIN_DETECTION_CONFIDENCE"): s4.MIN_DETECTION_CONFIDENCE = 0.5
    if hasattr(s4, "MIN_TRACKING_CONFIDENCE"): s4.MIN_TRACKING_CONFIDENCE = 0.5
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

def main():
    ensure_dirs()
    vids = sorted(INPUT_DIR.glob("*.mkv"))
    if not vids:
        print(f"⚠️ No .mkv files found in {INPUT_DIR.resolve()}")
        sys.exit(0)

    print("FRI Handover — Full Pipeline Runner (re-runs everything)")
    print(f"Found {len(vids)} video(s): {[v.name for v in vids]}")

    for v in vids:
        run_for_video(v)

if __name__ == "__main__":
    main()
