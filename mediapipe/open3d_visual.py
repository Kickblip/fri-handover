from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

import open3d as o3d
import mediapipe as mp


# ============================
# ====== CONFIG / DEFAULTS ===
# ============================

DEFAULT_HANDS_CSV = Path("dataset/input_video/batch-testing/2_w_b_hands.csv")
# Example: change this to your real *_vertices.csv path
DEFAULT_BOX_CSV = Path("/home/bwilab/Documents/fri-handover/dataset/input_video/batch-testing/2_w_b_box.csv")

# MediaPipe hand graph
mp_hands = mp.solutions.hands
HAND_CONNECTIONS = list(mp_hands.HAND_CONNECTIONS)

# Box edges (same as in your box scripts)
BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


# ============================
# ====== AXIS REMAP ==========
# ============================

def remap_coords_xyz(points: np.ndarray) -> np.ndarray:
    """
    Apply your axis mapping:
      Open3D X = X
      Open3D Y = Z
      Open3D Z = -Y
    Input:  (N,3) [x,y,z] in camera coords
    Output: (N,3) in remapped coords.
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    return np.stack([x, y, z], axis=1)


# ============================
# ====== HANDS HELPERS =======
# ============================

def load_hands_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "frame_idx" not in df.columns:
        raise ValueError("Hands CSV must have a 'frame' column.")
    return df


def extract_hand_points_for_frame(
    df: pd.DataFrame, frame_idx: int, hand_idx: int
) -> Tuple[Optional[np.ndarray], Optional[List[List[int]]]]:
    """
    Returns:
      points: (M,3) or None
      lines: list of [i,j] pairs (using indices into points) or None
    """
    row = df[df["frame_idx"] == frame_idx]
    if row.empty:
        return None, None

    row = row.iloc[0]
    coords = []
    for i in range(21):
        x = row.get(f"h{hand_idx}_lm{i}_x", np.nan)
        y = row.get(f"h{hand_idx}_lm{i}_y", np.nan)
        z = row.get(f"h{hand_idx}_lm{i}_z", np.nan)
        coords.append([x, y, z])

    pts = np.asarray(coords, dtype=float)
    valid_mask = ~np.isnan(pts).any(axis=1)
    if valid_mask.sum() == 0:
        return None, None

    # Remap axes
    pts = remap_coords_xyz(pts)

    # Build index map old -> new (only for valid landmarks)
    idx_map = {}
    valid_pts = []
    for old_idx, is_valid in enumerate(valid_mask):
        if is_valid:
            idx_map[old_idx] = len(valid_pts)
            valid_pts.append(pts[old_idx])

    valid_pts = np.asarray(valid_pts, dtype=float)

    # Build lines, only if both endpoints valid
    lines = []
    for a, b in HAND_CONNECTIONS:
        if a in idx_map and b in idx_map:
            lines.append([idx_map[a], idx_map[b]])

    if len(lines) == 0:
        return valid_pts, None

    return valid_pts, lines


# ============================
# ====== BOX HELPERS =========
# ============================

def load_box_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"frame_idx", "tag_id"}
    for i in range(8):
        required |= {f"v{i}_x", f"v{i}_y", f"v{i}_z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Box CSV missing columns: {sorted(missing)}")
    return df


def extract_box_points_for_frame(
    df: pd.DataFrame, frame_idx: int
) -> Optional[np.ndarray]:
    """
    Returns (8,3) vertices for this frame, or None.
    Chooses the first tag_id >= 0 if multiple detections exist.
    """
    sub = df[df["frame_idx"] == frame_idx]
    sub = sub[sub["tag_id"] >= 0]
    if sub.empty:
        return None

    row = sub.iloc[0]
    verts = np.zeros((8, 3), dtype=float)
    for i in range(8):
        x = row[f"v{i}_x"]
        y = row[f"v{i}_y"]
        z = row[f"v{i}_z"]
        verts[i] = [x, y, z-.2]

    # Might be all NaNs if no valid data
    if np.isnan(verts).all():
        return None

    verts = remap_coords_xyz(verts)
    return verts


# ============================
# ====== MAIN VISUALIZER =====
# ============================

def main():
    parser = argparse.ArgumentParser(
        description="Open3D viewer for 3D hands + AprilTag box over time."
    )
    parser.add_argument(
        "--hands-csv",
        type=str,
        default=str(DEFAULT_HANDS_CSV),
        help="CSV from depth_results.csv (3D hand landmarks)",
    )
    parser.add_argument(
        "--box-csv",
        type=str,
        default=str(DEFAULT_BOX_CSV),
        help="CSV from *_vertices.csv (AprilTag box vertices)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Playback FPS for animation",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on number of frames to play",
    )
    args = parser.parse_args()

    hands_df = load_hands_csv(Path(args.hands_csv))
    box_df = load_box_csv(Path(args.box_csv))

    max_hand_frame = int(hands_df["frame_idx"].max())
    max_box_frame = int(box_df["frame_idx"].max())
    total_frames = max(max_hand_frame, max_box_frame) + 1

    if args.max_frames is not None:
        total_frames = min(total_frames, args.max_frames)

    print(f"Total frames (hands/box): {total_frames}")

    # ----- Open3D setup -----
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Hands + Box (Open3D)", width=1280, height=720)

    # Coordinate frame at origin
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis.add_geometry(coord_frame)

    # Hand geometries (two hands max)
    hand_geoms: List[o3d.geometry.LineSet] = []
    for _ in range(2):
        ls = o3d.geometry.LineSet()
        hand_geoms.append(ls)
        vis.add_geometry(ls)

    # Box geometry
    box_geom = o3d.geometry.LineSet()
    vis.add_geometry(box_geom)

    # Initial camera viewpoint will auto-fit after first update
    first = True
    dt = 1.0 / max(args.fps, 1e-3)

    for frame_idx in range(total_frames):
        # ---------- Hands ----------
        for h in range(2):
            pts, lines = extract_hand_points_for_frame(hands_df, frame_idx, h)
            ls = hand_geoms[h]
            if pts is None or lines is None or len(pts) == 0:
                ls.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
                ls.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
                continue

            ls.points = o3d.utility.Vector3dVector(pts)
            ls.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
            # Color hands: h0 blue-ish, h1 red-ish
            color = np.array([[0.2, 0.4, 1.0]] if h == 0 else [[1.0, 0.3, 0.3]])
            ls.colors = o3d.utility.Vector3dVector(
                np.repeat(color, len(lines), axis=0)
            )

        # ---------- Box ----------
        verts = extract_box_points_for_frame(box_df, frame_idx)
        if verts is None:
            box_geom.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            box_geom.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
        else:
            box_geom.points = o3d.utility.Vector3dVector(verts)
            box_geom.lines = o3d.utility.Vector2iVector(
                np.asarray(BOX_EDGES, dtype=np.int32)
            )
            box_geom.colors = o3d.utility.Vector3dVector(
                np.tile(np.array([[0.0, 1.0, 0.0]]), (len(BOX_EDGES), 1))
            )

        # ---------- Render ----------
        for g in hand_geoms:
            vis.update_geometry(g)
        vis.update_geometry(box_geom)
        vis.update_geometry(coord_frame)

        if first:
            vis.reset_view_point(True)
            first = False

        vis.poll_events()
        vis.update_renderer()

        time.sleep(dt)

    print("Animation finished. You can still interact with the window; close it to exit.")
    # Keep window open until user closes
    while True:
        if not vis.poll_events():
            break
        vis.update_renderer()
        time.sleep(0.03)

    vis.destroy_window()


if __name__ == "__main__":
    main()
