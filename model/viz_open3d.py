"""
Open3D visualizer (internally OpenGL) for qualitative validation.

It reads:
  - dataset/mediapipe_outputs/csv/world/{stem}_world.csv       (3D hand landmarks)
  - dataset/vertices_csv/{stem}_vertices.csv                    (object vertices)
  - dataset/mediapipe_outputs/csv/rodrigues/{stem}_probs.csv    (model per-frame probs)

…and renders a 3D animation where color encodes handover probability:
  blue (low) -> red (high). The object brightens with higher probability.

Usage:
  python viz_open3d.py 1_video
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import open3d as o3d  # uses OpenGL under the hood

# ------------------ Paths ------------------
ROOT = Path("dataset")
WORLD_DIR = ROOT / "mediapipe_outputs" / "csv" / "world"
VERT_DIR  = ROOT / "vertices_csv"
PROB_DIR  = ROOT / "mediapipe_outputs" / "csv" / "rodrigues"

# ------------------ Loaders ------------------
def load_world_points(stem: str):
    """
    Load 3D hand landmarks per frame. Supports:
      (A) long-form: columns [frame, hand, landmark?, x, y, z]
      (B) wide-form: columns [frame, x0, y0, z0, x1, y1, z1, ...]
    Returns:
      frames: [T] list of frame indices
      per_frame_points: list of (N,3) float arrays (N = #points that frame)
    """
    p = WORLD_DIR / f"{stem}_world.csv"
    df = pd.read_csv(p)

    # Long form (preferred)
    if {"frame", "hand", "x", "y", "z"}.issubset(df.columns):
        frames = sorted(df["frame"].unique().tolist())
        out = []
        for f in frames:
            sub = df[df["frame"] == f][["x", "y", "z"]].to_numpy(dtype=np.float32)
            out.append(sub)
        return frames, out

    # Wide form (fall back): assume consecutive triplets = (x,y,z)
    if "frame" in df.columns:
        frames = df["frame"].astype(int).tolist()
        cols = [c for c in df.columns if c != "frame"]
        arr = df[cols].to_numpy(dtype=np.float32)
        n_pts = len(cols) // 3
        out = [arr[i].reshape(n_pts, 3) for i in range(arr.shape[0])]
        return frames, out

    raise ValueError("Unrecognized 'world' CSV schema; need long or wide format.")

def load_vertices(stem: str, frames):
    """
    Load object vertices per frame, aligned to 'frames'.
    If a frame is missing in the CSV, fill with empty array for that frame.
    """
    p = VERT_DIR / f"{stem}_vertices.csv"
    if not p.exists():
        # Gracefully handle missing vertices file
        return [np.zeros((0, 3), np.float32) for _ in frames]

    df = pd.read_csv(p)
    if "frame" not in df.columns:
        raise ValueError(f"{p} must include a 'frame' column.")

    feat_cols = [c for c in df.columns if c != "frame"]
    D = len(feat_cols)
    rows = {int(r["frame"]): r[feat_cols].to_numpy(dtype=np.float32) for _, r in df.iterrows()}

    out = []
    for f in frames:
        vec = rows.get(f)
        if vec is None or len(vec) == 0:
            out.append(np.zeros((0, 3), np.float32))
            continue
        # Ensure length divisible by 3 for (x,y,z) grouping
        if D % 3 != 0:
            pad = 3 - (D % 3)
            vec = np.pad(vec, (0, pad))
        out.append(vec.reshape(-1, 3))
    return out

def load_probs(stem: str):
    """
    Load per-frame probabilities saved by `python -m handover.infer --out`.
    Returns a dict: frame -> prob.
    """
    p = PROB_DIR / f"{stem}_probs.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing probs CSV: {p}\n"
                                f"Run inference first, e.g.: "
                                f"python -m handover.infer {stem} --out {p}")
    df = pd.read_csv(p)
    if not {"frame", "prob"}.issubset(df.columns):
        raise ValueError(f"{p} must have columns 'frame,prob'.")
    return {int(r["frame"]): float(r["prob"]) for _, r in df.iterrows()}

# ------------------ Color mapping ------------------
def prob_to_color(prob: float):
    """
    Map probability in [0,1] to RGB color:
      0.0 -> blue (0,0,1)
      1.0 -> red  (1,0,0)
    """
    return np.array([prob, 0.0, 1.0 - prob], dtype=np.float32)

# ------------------ Visualization loop ------------------
def main(stem: str):
    # Load geometry + predictions
    frames, hand_pts = load_world_points(stem)
    verts_pts = load_vertices(stem, frames)
    probs = load_probs(stem)

    # Prepare point clouds
    hand_pc = o3d.geometry.PointCloud()  # hands
    obj_pc  = o3d.geometry.PointCloud()  # object (baton vertices)

    # Create window (OpenGL context under the hood)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"FRI Handover — {stem}", width=1280, height=720)

    # Register geometry with the scene
    vis.add_geometry(hand_pc)
    vis.add_geometry(obj_pc)

    # Tweak rendering (point size, background)
    opt = vis.get_render_option()
    opt.point_size = 4.0
    opt.background_color = np.array([0.0, 0.0, 0.0])

    # Playback over frames: update point positions + colors and re-render
    for i, f in enumerate(frames):
        # --- hands ---
        hp = hand_pts[i].astype(np.float32) if len(hand_pts[i]) else np.zeros((0, 3), np.float32)
        hand_pc.points = o3d.utility.Vector3dVector(hp)

        pr = probs.get(f, 0.0)  # probability for this frame
        hc = np.tile(prob_to_color(pr), (len(hp), 1)) if len(hp) else np.zeros((0, 3), np.float32)
        hand_pc.colors = o3d.utility.Vector3dVector(hc)

        # --- object ---
        vp = verts_pts[i].astype(np.float32) if len(verts_pts[i]) else np.zeros((0, 3), np.float32)
        obj_pc.points = o3d.utility.Vector3dVector(vp)

        # bright yellow whose brightness scales with prob (0.3..1.0)
        yellow = np.array([1.0, 1.0, 0.0], dtype=np.float32)
        oc = np.tile(yellow * (0.3 + 0.7 * pr), (len(vp), 1)) if len(vp) else np.zeros((0, 3), np.float32)
        obj_pc.colors = o3d.utility.Vector3dVector(oc)

        # push updates to the renderer
        vis.update_geometry(hand_pc)
        vis.update_geometry(obj_pc)
        vis.poll_events()
        vis.update_renderer()

    # Let user interact before closing (orbit, zoom, etc.)
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python viz_open3d.py <stem>   # e.g., 1_video")
        sys.exit(1)
    main(sys.argv[1])
