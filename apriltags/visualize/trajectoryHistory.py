import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from matplotlib import cm, colors as mcolors

# ---- geometry ----
BOX_EDGES = [(0,1),(1,2),(2,3),(3,0),
             (4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]

BOX_FACES = [
    [0,1,2,3],
    [4,5,6,7],
    [0,1,5,4],
    [1,2,6,5],
    [2,3,7,6],
    [3,0,4,7]
]

AXIS_NAMES = ["x", "y", "z"]

# ---- axis remap helpers ----
def _parse_axes(spec: str):
    parts = [p.strip().lower() for p in spec.split(",")]
    if len(parts) != 3:
        raise ValueError("axes spec must have 3 comma-separated items, e.g. 'x,-y,z'")
    perm, signs, used = [], [], set()
    for p in parts:
        sgn = -1 if p.startswith("-") else 1
        name = p.lstrip("+-")
        if name not in AXIS_NAMES:
            raise ValueError(f"Invalid axis '{p}'. Use x,y,z with optional leading '-'")
        idx = AXIS_NAMES.index(name)
        if idx in used:
            raise ValueError("Duplicate axis in spec")
        used.add(idx)
        perm.append(idx)
        signs.append(sgn)
    return perm, signs

def apply_axes_remap(verts_Tx8x3, perm, signs):
    v = verts_Tx8x3.copy()
    M = np.zeros((3, 3), dtype=float)
    for new_i, (old_i, s) in enumerate(zip(perm, signs)):
        M[new_i, old_i] = float(s)
    flat = v.reshape(-1, 3)
    flat = (M @ flat.T).T
    return flat.reshape(v.shape)

# ---- data loading ----
def choose_tag_id(df, user_tag):
    if user_tag is not None:
        return user_tag
    tags = df.loc[df["tag_id"] >= 0, "tag_id"]
    if tags.empty:
        return None
    return int(tags.mode().iloc[0])

def load_vertices(csv_path, tag_id, interp):
    df = pd.read_csv(csv_path)
    required = {"frame_idx","tag_id"}
    for i in range(8):
        required |= {f"v{i}_x", f"v{i}_y", f"v{i}_z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    sel_tag = choose_tag_id(df, tag_id)
    if sel_tag is None:
        raise ValueError("No valid tag rows (all tag_id=-1).")

    sub = df[df["tag_id"] == sel_tag].copy()
    if sub.empty:
        raise ValueError(f"No rows for tag_id={sel_tag}")
    sub = sub.sort_values("frame_idx")

    full_idx = np.arange(sub["frame_idx"].min(), sub["frame_idx"].max()+1, dtype=int)
    sub = sub.set_index("frame_idx").reindex(full_idx)

    T = len(sub)
    verts = np.full((T, 8, 3), np.nan, dtype=float)
    for vi in range(8):
        verts[:, vi, 0] = sub[f"v{vi}_x"].to_numpy()
        verts[:, vi, 1] = sub[f"v{vi}_y"].to_numpy()
        verts[:, vi, 2] = sub[f"v{vi}_z"].to_numpy()

    if interp:
        t = np.arange(T, dtype=float)
        for vi in range(8):
            for ax in range(3):
                y = verts[:, vi, ax]
                m = ~np.isnan(y)
                if m.sum() >= 2:
                    verts[:, vi, ax] = np.interp(t, t[m], y[m])

    return sel_tag, full_idx, verts  # (T,8,3)

# ---- bounds ----
def global_bounds(verts_Tx8x3, pad=0.05):
    xyz = verts_Tx8x3.reshape(-1, 3)
    xyz = xyz[~np.isnan(xyz).any(axis=1)]
    if xyz.size == 0:
        raise ValueError("No finite vertex data to render.")
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = (mins + maxs) / 2.0
    span = (maxs - mins)
    max_range = max(span.max(), 1e-6)
    half = max_range * (0.5 + pad)
    lims = np.vstack([center - half, center + half])
    return lims  # (2,3)

def build_collections_for_frame(v8x3):
    segs = [[v8x3[a], v8x3[b]] for (a,b) in BOX_EDGES]
    faces = [[v8x3[i] for i in idxs] for idxs in BOX_FACES]
    return segs, faces

# ---- main ----
def main():
    ap = argparse.ArgumentParser(description="Render a single 3D overlay image of the box across all timesteps.")
    ap.add_argument("csv_path", type=str, help="Path to *_vertices.csv")
    ap.add_argument("--out", type=str, default=None, help="Output PNG path (default: <csv>_overlay.png)")
    ap.add_argument("--width", type=int, default=1280, help="Output width in pixels")
    ap.add_argument("--height", type=int, default=720, help="Output height in pixels")
    ap.add_argument("--dpi", type=int, default=100, help="Matplotlib DPI (controls figure sizing)")
    ap.add_argument("--elev", type=float, default=15.0, help="Camera elevation (deg)")
    ap.add_argument("--azim", type=float, default=-60.0, help="Camera azimuth (deg)")
    ap.add_argument("--tag-id", type=int, default=None, help="Select a specific tag_id (default: auto)")
    ap.add_argument("--interp", action="store_true", help="Interpolate missing frames")
    ap.add_argument("--axes", type=str, default="x,z,-y",
                    help="Axis remap like 'x,-y,z' (new axes expressed in terms of old).")
    ap.add_argument("--stride", type=int, default=1, help="Use every Nth frame to reduce clutter")
    ap.add_argument("--max-frames", type=int, default=None, help="Cap the number of frames drawn (after stride)")
    ap.add_argument("--cmap", type=str, default="viridis", help="Matplotlib colormap for time coloring")
    ap.add_argument("--edge-alpha", type=float, default=0.35, help="Alpha for edges per frame")
    ap.add_argument("--face-alpha", type=float, default=0.08, help="Alpha for faces per frame")
    ap.add_argument("--draw-faces", action="store_true", help="Overlay translucent faces (can get heavy)")
    ap.add_argument("--draw-points", action="store_true", help="Draw vertex scatter points per frame")
    args = ap.parse_args()

    if args.out is None:
        base = os.path.splitext(os.path.abspath(args.csv_path))[0]
        args.out = base + "_overlay.png"

    # Load & remap
    sel_tag, frame_idx, verts = load_vertices(args.csv_path, args.tag_id, args.interp)
    perm, signs = _parse_axes(args.axes)
    verts = apply_axes_remap(verts, perm, signs)

    # Frame selection
    T = verts.shape[0]
    all_ids = np.arange(T)
    keep = all_ids[::max(1, args.stride)]
    if args.max_frames is not None:
        keep = keep[:args.max_frames]

    # Bounds
    lims = global_bounds(verts[keep], pad=0.08)
    mins, maxs = lims[0], lims[1]

    # Figure sizing (inches = pixels/dpi)
    fig_w = args.width / args.dpi
    fig_h = args.height / args.dpi
    plt.rcParams["figure.dpi"] = args.dpi
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_subplot(111, projection="3d")

    # Labels show mapping
    label_parts = []
    for new_name, old_idx, s in zip(["X","Y","Z"], perm, signs):
        sign_txt = "" if s > 0 else "-"
        label_parts.append(f"{new_name} = {sign_txt}{AXIS_NAMES[old_idx].upper()}")

    ax.set_title(f"Box Over Time (tag_id={sel_tag})")
    ax.set_xlabel(f"X (m)  [{label_parts[0]}]")
    ax.set_ylabel(f"Y (m)  [{label_parts[1]}]")
    ax.set_zlabel(f"Z (m)  [{label_parts[2]}]")

    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    try:
        ax.set_box_aspect((1,1,1))
    except Exception:
        pass
    ax.view_init(elev=args.elev, azim=args.azim)

    # Color by time
    cmap = cm.get_cmap(args.cmap)
    norm = mcolors.Normalize(vmin=keep.min() if len(keep) else 0,
                             vmax=keep.max() if len(keep) else 1)

    # Accumulate all edge segments (faster as one collection)
    all_segs = []
    seg_colors = []

    # Optional faces: collect then add one Poly3DCollection
    all_faces = []
    face_colors = []

    # Optional points
    px, py, pz, pcolors = [], [], [], []

    last_valid = None
    for i in keep:
        v = verts[i]
        if np.isnan(v).any():
            # hold last valid if available
            if last_valid is None:
                continue
            v = last_valid
        else:
            last_valid = v

        segs, faces = build_collections_for_frame(v)
        col = list(cmap(norm(i)))
        # edges alpha
        col[3] = np.clip(args.edge_alpha, 0.0, 1.0)
        for _ in segs:
            all_segs.append(_)
            seg_colors.append(tuple(col))

        if args.draw_faces:
            fcol = list(cmap(norm(i)))
            fcol[3] = np.clip(args.face_alpha, 0.0, 1.0)
            for f in faces:
                all_faces.append(f)
                face_colors.append(tuple(fcol))

        if args.draw_points:
            px.extend(v[:,0]); py.extend(v[:,1]); pz.extend(v[:,2])
            pcolors.extend([tuple(col)]*8)

    # Draw one shot
    if all_segs:
        edge_coll = Line3DCollection(all_segs, linewidths=1.5, colors=seg_colors)
        ax.add_collection3d(edge_coll)
    if args.draw_faces and all_faces:
        face_coll = Poly3DCollection(all_faces, facecolors=face_colors, linewidths=0.0)
        ax.add_collection3d(face_coll)
    if args.draw_points and len(px) > 0:
        ax.scatter(px, py, pz, s=6, c=pcolors, depthshade=False)

    # Save at exact pixel size
    fig.set_size_inches(fig_w, fig_h, forward=True)
    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    mapping_str = ", ".join(label_parts)
    print(f"Saved overlay image: {args.out}  ({args.width}x{args.height} @ {args.dpi} dpi)")
    print("Base coords (input): COLOR CAMERA frame (OpenCV) => +X right, +Y down, +Z forward; units: meters.")
    print(f"Applied axis mapping: {mapping_str}")
    print(f"Frames drawn: {len(keep)} (stride={args.stride}, max_frames={args.max_frames})")

if __name__ == "__main__":
    main()
