"""
Open3D (OpenGL) viewer:
- Colors hand points blue→red by probability.
- Brightens object with higher probability.
- Reads predictions from dataset/model_output/predictions/<stem>_probs.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import open3d as o3d

ROOT = Path("dataset")
WORLD_DIR = ROOT / "mediapipe_outputs" / "csv" / "world"
VERT_DIR  = ROOT / "vertices_csv"
PRED_DIR  = ROOT / "model_output" / "predictions"

def _pick_col(df, main, aliases):
    for c in [main]+aliases:
        if c in df.columns: return c
    raise KeyError(f"Need one of {[main]+aliases} in {list(df.columns)[:8]}")

def load_world_points(stem: str):
    """
    Wide format support: use all *_world_{x,y,z}_{0/1} columns present.
    """
    p = WORLD_DIR / f"{stem}_world.csv"
    df = pd.read_csv(p)
    fcol = _pick_col(df, "frame_index", ["frame","frame_idx"])
    frames = df[fcol].astype(int).tolist()
    cols = [c for c in df.columns if c.endswith("_world_x_0") or c.endswith("_world_y_0") or c.endswith("_world_z_0")
                                      or c.endswith("_world_x_1") or c.endswith("_world_y_1") or c.endswith("_world_z_1")]
    arr = df[cols].to_numpy(np.float32) if cols else np.zeros((len(frames), 0), np.float32)
    n_pts = (arr.shape[1] // 3) if arr.size else 0
    per = [arr[i].reshape(n_pts, 3) if n_pts else np.zeros((0,3), np.float32) for i in range(len(frames))]
    return frames, per

def load_vertices(stem: str, frames):
    p = VERT_DIR / f"{stem}_vertices.csv"
    if not p.exists():
        return [np.zeros((0,3), np.float32) for _ in frames]
    df = pd.read_csv(p)
    fcol = _pick_col(df, "frame", ["frame_index","frame_idx"])
    cols = [c for c in df.columns if c != fcol]
    rows = {int(r[fcol]): r[cols].to_numpy(np.float32) for _, r in df.iterrows()}
    out = []
    for f in frames:
        vec = rows.get(f)
        if vec is None or len(vec)==0:
            out.append(np.zeros((0,3), np.float32)); continue
        if len(vec)%3 != 0:
            vec = np.pad(vec, (0, 3 - len(vec)%3))
        out.append(vec.reshape(-1,3))
    return out

def load_probs(stem: str):
    p = PRED_DIR / f"{stem}_probs.csv"
    df = pd.read_csv(p)
    fcol = _pick_col(df, "frame", ["frame_index","frame_idx"])
    return {int(r[fcol]): float(r["prob"]) for _, r in df.iterrows()}

def prob_to_color(p: float):  # 0 -> blue, 1 -> red
    return np.array([p, 0.0, 1.0 - p], np.float32)

def main(stem: str):
    frames, hand_pts = load_world_points(stem)
    verts_pts = load_vertices(stem, frames)
    probs = load_probs(stem)

    hand_pc = o3d.geometry.PointCloud()
    obj_pc  = o3d.geometry.PointCloud()
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"FRI Handover — {stem}", width=1280, height=720)
    vis.add_geometry(hand_pc); vis.add_geometry(obj_pc)
    opt = vis.get_render_option(); opt.point_size = 4.0; opt.background_color = np.array([0,0,0], np.float32)

    for i, f in enumerate(frames):
        # hands
        hp = hand_pts[i] if hand_pts[i].size else np.zeros((0,3), np.float32)
        hand_pc.points = o3d.utility.Vector3dVector(hp)
        pr = float(probs.get(f, 0.0))
        hc = np.tile(prob_to_color(pr), (len(hp),1)) if len(hp) else np.zeros((0,3), np.float32)
        hand_pc.colors = o3d.utility.Vector3dVector(hc)

        # object
        vp = verts_pts[i] if len(verts_pts[i]) else np.zeros((0,3), np.float32)
        obj_pc.points  = o3d.utility.Vector3dVector(vp)
        yellow = np.array([1.0,1.0,0.0], np.float32)
        oc = np.tile(yellow*(0.3+0.7*pr), (len(vp),1)) if len(vp) else np.zeros((0,3), np.float32)
        obj_pc.colors = o3d.utility.Vector3dVector(oc)

        vis.update_geometry(hand_pc); vis.update_geometry(obj_pc)
        vis.poll_events(); vis.update_renderer()

    vis.run(); vis.destroy_window()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualization/viz_open3d.py <stem>")
        sys.exit(1)
    main(sys.argv[1])
