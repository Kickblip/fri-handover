"""
Open3D viewer to compare actual vs predicted receiving-hand landmarks.

Usage:
    python -m model.viz_compare_open3d <stem> [--pred path/to/predictions.csv]

Inputs:
    - Actual landmarks: dataset/mediapipe_outputs/csv/world/{stem}_world.csv
      (expects *_world_{x,y,z}_1 columns for the receiving hand)
    - Predictions: dataset/model_output/predictions/{stem}_future_predictions.csv
      (created by `python -m model.infer <stem> --video`)

Visualization:
    - Actual hand landmarks   → GREEN points
    - Predicted hand landmarks → RED points
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import open3d as o3d
import pandas as pd

from .config import ROOT

WORLD_DIR = ROOT / "mediapipe_outputs" / "csv" / "world"
PRED_DIR = ROOT / "model_output" / "predictions"


def _pick_col(df: pd.DataFrame, main: str, aliases: list[str]) -> str:
    for c in [main] + aliases:
        if c in df.columns:
            return c
    raise KeyError(f"Need one of {[main] + aliases} in {df.columns[:8].tolist()}...")


def _detect_hand_suffix(df: pd.DataFrame) -> str:
    """Return the suffix (_0 or _1) present in world coordinate columns."""
    for suffix in ["_1", "_0"]:
        cols = [
            c
            for c in df.columns
            if c.endswith(f"_world_x{suffix}")
            or c.endswith(f"_world_y{suffix}")
            or c.endswith(f"_world_z{suffix}")
        ]
        if cols:
            return suffix
    raise ValueError(
        "Could not find columns ending with '_world_[xyz]_0' or '_world_[xyz]_1'."
    )


def load_actual_receiving_hand(stem: str) -> Tuple[Dict[int, np.ndarray], list[int]]:
    """Load actual receiving-hand world coordinates (21×3 per frame)."""
    world_path = WORLD_DIR / f"{stem}_world.csv"
    if not world_path.exists():
        raise FileNotFoundError(f"Missing world CSV: {world_path}")

    df = pd.read_csv(world_path)
    fcol = _pick_col(df, "frame", ["frame_index", "frame_idx"])
    frames = df[fcol].astype(int).tolist()

    suffix = _detect_hand_suffix(df)
    cols = sorted(
        c
        for c in df.columns
        if c.endswith(f"_world_x{suffix}")
        or c.endswith(f"_world_y{suffix}")
        or c.endswith(f"_world_z{suffix}")
    )
    if not cols:
        raise ValueError(
            f"No receiving-hand columns ending with '_world_[xyz]{suffix}' found in {world_path}"
        )

    arr = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    arr = np.nan_to_num(arr, nan=0.0)
    n_landmarks = arr.shape[1] // 3
    per_frame = arr.reshape(len(frames), n_landmarks, 3)
    return {frame: per_frame[i] for i, frame in enumerate(frames)}, frames


def load_predictions(stem: str, csv_path: Path | None = None) -> Dict[int, np.ndarray]:
    """Load predicted future frames and map to actual target frame index."""
    if csv_path is None:
        csv_path = PRED_DIR / f"{stem}_future_predictions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing predictions CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"frame", "future_frame_idx"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Predictions CSV must contain {required}, got {df.columns[:8].tolist()}"
        )

    lm_cols = [c for c in df.columns if c.startswith("lm_") and c.endswith(("_x", "_y", "_z"))]
    if not lm_cols:
        raise ValueError(f"No landmark columns (lm_*_{x|y|z}) found in {csv_path}")

    lm_cols = sorted(lm_cols)
    arr = df[lm_cols].to_numpy(np.float32)
    arr = np.nan_to_num(arr, nan=0.0)
    n_landmarks = arr.shape[1] // 3

    pred_map: Dict[int, Tuple[np.ndarray, int]] = {}
    for row, coords in zip(df.itertuples(index=False), arr):
        base_frame = int(row.frame)
        horizon = int(row.future_frame_idx)
        target_frame = base_frame + horizon + 1  # future frame relative to base
        pts = coords.reshape(n_landmarks, 3)

        # Keep the prediction generated from the most recent base frame
        if target_frame in pred_map and base_frame <= pred_map[target_frame][1]:
            continue
        pred_map[target_frame] = (pts, base_frame)

    return {frame: data[0] for frame, data in pred_map.items()}


def visualize(stem: str, pred_map: Dict[int, np.ndarray], actual_map: Dict[int, np.ndarray]):
    shared_frames = sorted(set(pred_map.keys()) & set(actual_map.keys()))
    if not shared_frames:
        raise RuntimeError("No overlapping frames between predictions and ground truth.")

    actual_pc = o3d.geometry.PointCloud()
    pred_pc = o3d.geometry.PointCloud()

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Handover Compare — {stem}", width=1280, height=720)
    vis.add_geometry(actual_pc)
    vis.add_geometry(pred_pc)
    render_opt = vis.get_render_option()
    render_opt.point_size = 6.0
    render_opt.background_color = np.array([0, 0, 0], np.float32)

    green = np.array([0.0, 1.0, 0.0], np.float32)
    red = np.array([1.0, 0.0, 0.0], np.float32)

    for frame in shared_frames:
        actual = actual_map[frame]
        pred = pred_map[frame]

        actual_pc.points = o3d.utility.Vector3dVector(actual)
        actual_pc.colors = o3d.utility.Vector3dVector(
            np.tile(green, (len(actual), 1))
        )

        pred_pc.points = o3d.utility.Vector3dVector(pred)
        pred_pc.colors = o3d.utility.Vector3dVector(np.tile(red, (len(pred), 1)))

        vis.update_geometry(actual_pc)
        vis.update_geometry(pred_pc)
        vis.poll_events()
        vis.update_renderer()
    vis.run()
    vis.destroy_window()


def main():
    ap = argparse.ArgumentParser(description="Visualize actual vs predicted hand landmarks in 3D.")
    ap.add_argument("stem", help="Video/data stem, e.g., 1_w_b")
    ap.add_argument(
        "--pred",
        type=Path,
        default=None,
        help="Optional custom predictions CSV (defaults to dataset/model_output/predictions/<stem>_future_predictions.csv)",
    )
    args = ap.parse_args()

    actual_map, _ = load_actual_receiving_hand(args.stem)
    pred_map = load_predictions(args.stem, args.pred)
    visualize(args.stem, pred_map, actual_map)


if __name__ == "__main__":
    main()

