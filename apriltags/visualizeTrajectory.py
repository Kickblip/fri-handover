import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from matplotlib.animation import FuncAnimation, FFMpegWriter


# script for generating cool animations of the box's vertices, made with chatgpt

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

def _parse_axes(spec: str):
    """
    Parse an axis remap spec like 'x,-y,z' or 'x,z,-y'.
    Returns (perm, signs), where:
      - perm[new_axis_index] = old_axis_index
      - signs[new_axis_index] = +1 or -1
    """
    parts = [p.strip().lower() for p in spec.split(",")]
    if len(parts) != 3:
        raise ValueError("axes spec must have 3 comma-separated items, e.g. 'x,-y,z'")

    perm = []
    signs = []
    used = set()
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
    """
    Apply axis remap to verts with shape (T, 8, 3).
    new = M @ old, where M is built from perm/signs.
    """
    v = verts_Tx8x3.copy()
    M = np.zeros((3, 3), dtype=float)
    for new_i, (old_i, s) in enumerate(zip(perm, signs)):
        M[new_i, old_i] = float(s)

    flat = v.reshape(-1, 3)              # (T*8, 3)
    flat = (M @ flat.T).T                # apply transform
    return flat.reshape(v.shape)

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

    # Build contiguous timeline so playback is smooth
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

def global_bounds(verts_Tx8x3, pad=0.05):
    """Compute global min/max across all frames; add small padding."""
    xyz = verts_Tx8x3.reshape(-1, 3)
    xyz = xyz[~np.isnan(xyz).any(axis=1)]
    if xyz.size == 0:
        raise ValueError("No finite vertex data to render.")
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = (mins + maxs) / 2.0
    span = (maxs - mins)
    # Enforce a cube to avoid axis distortion
    max_range = max(span.max(), 1e-6)
    half = max_range * (0.5 + pad)  # a bit of margin
    lims = np.vstack([center - half, center + half])
    return lims  # shape (2,3) -> [mins, maxs]

def build_collections_for_frame(v8x3):
    segs = [[v8x3[a], v8x3[b]] for (a,b) in BOX_EDGES]
    faces = [[v8x3[i] for i in idxs] for idxs in BOX_FACES]
    return segs, faces

def main():
    ap = argparse.ArgumentParser(description="Animate reconstructed 3D box and save as MP4.")
    ap.add_argument("csv_path", type=str, help="Path to *_vertices.csv")
    ap.add_argument("--out", type=str, default=None, help="Output MP4 path (default: <csv>_anim.mp4)")
    ap.add_argument("--fps", type=int, default=30, help="Frames per second for output video")
    ap.add_argument("--width", type=int, default=1280, help="Output width in pixels")
    ap.add_argument("--height", type=int, default=720, help="Output height in pixels")
    ap.add_argument("--dpi", type=int, default=100, help="Matplotlib DPI (controls figure sizing)")
    ap.add_argument("--elev", type=float, default=15.0, help="Camera elevation (deg)")
    ap.add_argument("--azim", type=float, default=-60.0, help="Camera azimuth (deg)")
    ap.add_argument("--tag-id", type=int, default=None, help="Select a specific tag_id (default: auto)")
    ap.add_argument("--interp", action="store_true", help="Interpolate missing frames")
    ap.add_argument("--bitrate", type=int, default=8000, help="FFmpeg bitrate (kbps)")
    ap.add_argument(
        "--axes",
        type=str,
        default="x,z,-y",
        help="Axis remap like 'x,-y,z' (new axes expressed in terms of old). "
             "Default keeps OpenCV camera: +X right, +Y down, +Z forward."
    )
    args = ap.parse_args()

    if args.out is None:
        base = os.path.splitext(os.path.abspath(args.csv_path))[0]
        args.out = base + "_anim.mp4"

    # Load vertices
    sel_tag, frame_idx, verts = load_vertices(args.csv_path, args.tag_id, args.interp)

    # === Apply axis remap BEFORE computing bounds (so limits/aspect match final coords) ===
    perm, signs = _parse_axes(args.axes)
    verts = apply_axes_remap(verts, perm, signs)

    T = verts.shape[0]

    # Compute global cubic bounds and aspect
    lims = global_bounds(verts, pad=0.08)  # add margin
    mins, maxs = lims[0], lims[1]
    box_aspect = (1, 1, 1)  # equal scaling in x,y,z

    # Figure sizing: inches = pixels / dpi
    fig_w = args.width / args.dpi
    fig_h = args.height / args.dpi
    plt.rcParams["figure.dpi"] = args.dpi
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_subplot(111, projection="3d")

    # Build pretty axis-label mapping text
    label_parts = []
    for new_name, old_idx, s in zip(["X","Y","Z"], perm, signs):
        sign_txt = "" if s > 0 else "-"
        label_parts.append(f"{new_name} = {sign_txt}{AXIS_NAMES[old_idx].upper()}")

    # Static axes setup
    ax.set_title(f"Box Animation (tag_id={sel_tag})")
    ax.set_xlabel(f"X (m)  [{label_parts[0]}]")
    ax.set_ylabel(f"Y (m)  [{label_parts[1]}]")
    ax.set_zlabel(f"Z (m)  [{label_parts[2]}]")
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    # Equal aspect (prevents "scrunched" distortion):
    try:
        ax.set_box_aspect(box_aspect)
    except Exception:
        pass
    ax.view_init(elev=args.elev, azim=args.azim)

    # Initial draw
    v0 = verts[0]
    segs0, faces0 = build_collections_for_frame(v0 if not np.isnan(v0).any() else np.zeros((8,3)))
    edge_coll = Line3DCollection(segs0, linewidths=2.0)
    face_coll = Poly3DCollection(faces0, alpha=0.18)
    scat = ax.scatter([], [], [], s=12)
    ax.add_collection3d(edge_coll)
    ax.add_collection3d(face_coll)
    txt = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    # Update function
    def update(i):
        v = verts[i]
        if np.isnan(v).any():
            # Hold last valid frame if available, else clear
            j = i - 1
            while j >= 0 and np.isnan(verts[j]).any():
                j -= 1
            if j >= 0:
                v = verts[j]
            else:
                edge_coll.set_segments([])
                face_coll.set_verts([])
                scat._offsets3d = ([], [], [])
                txt.set_text(f"frame {frame_idx[i]} (no data)")
                return edge_coll, face_coll, scat, txt

        segs, faces = build_collections_for_frame(v)
        edge_coll.set_segments(segs)
        face_coll.set_verts(faces)
        scat._offsets3d = (v[:,0], v[:,1], v[:,2])
        txt.set_text(f"frame {frame_idx[i]}")
        return edge_coll, face_coll, scat, txt

    interval_ms = int(1000 / max(1, args.fps))
    anim = FuncAnimation(fig, update, frames=T, interval=interval_ms, blit=False)

    # MP4 only
    try:
        writer = FFMpegWriter(fps=args.fps, bitrate=args.bitrate)
    except Exception as e:
        raise RuntimeError(
            "FFmpeg not found. Please install ffmpeg and ensure it is on your PATH."
        ) from e

    # Save at the exact pixel size by locking figure size & dpi
    fig.set_size_inches(fig_w, fig_h, forward=True)
    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    anim.save(args.out, writer=writer, dpi=args.dpi, savefig_kwargs={"bbox_inches": "tight"})
    plt.close(fig)

    # Print final info, including mapping
    mapping_str = ", ".join(label_parts)
    print(f"Saved MP4: {args.out}  ({args.width}x{args.height} @ {args.fps} fps)")
    print("Base coords (input): COLOR CAMERA frame (OpenCV) => +X right, +Y down, +Z forward; units: meters.")
    print(f"Applied axis mapping: {mapping_str}")

if __name__ == "__main__":
    main()
