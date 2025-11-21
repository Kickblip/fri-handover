from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from pyk4a import PyK4APlayback, CalibrationType

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


# ============================
# ==== PATHS (YOUR SETUP) ====
# ============================

input_video = Path(
    "/home/bwilab/Documents/fri-handover/record/1_video.mkv"
)
output_csv = Path("dataset/mediapipe_outputs/csv/depth_results.csv")
output_3d_video = Path("dataset/mediapipe_outputs/video/3d_visualization_output.mp4")
output_overlay_video = Path("dataset/mediapipe_outputs/video/3d_overlay_output.mp4")


# ============================
# ==== MEDIAPIPE HANDS =======
# ============================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1,
)

HAND_CONNECTIONS = list(mp_hands.HAND_CONNECTIONS)


# ============================
# ==== DEPTH → 3D UTILS ======
# ============================

def sample_depth_safely(depth: np.ndarray, u: int, v: int, radius: int = 1) -> float:
    """
    Take a small neighborhood around (u, v) and return the median
    of valid (>0) depth values. Greatly reduces spiky depth noise.
    """
    if depth is None:
        return 0.0

    h, w = depth.shape
    u0, u1 = max(0, u - radius), min(w, u + radius + 1)
    v0, v1 = max(0, v - radius), min(h, v + radius + 1)

    patch = depth[v0:v1, u0:u1]
    valid = patch[patch > 0]
    if valid.size == 0:
        return 0.0
    return float(np.median(valid))


def to_3d(calib, u: int, v: int, depth_mm: float) -> np.ndarray:
    """
    Convert pixel coordinate + depth to 3D using Azure Kinect calibration.
    Returns (x,y,z) in meters. If invalid, returns NaNs.
    """
    if depth_mm <= 0:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float32)

    try:
        x_mm, y_mm, z_mm = calib.convert_2d_to_3d(
            (u, v),
            float(depth_mm),
            CalibrationType.COLOR,
            CalibrationType.COLOR,
        )
        return np.array([x_mm, y_mm, z_mm], dtype=np.float32) / 1000.0  # mm → m
    except Exception:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float32)


# ============================
# ==== EXTRACT + OVERLAY =====
# ============================

def extract_3d_landmarks_and_overlay() -> Tuple[List[List[np.ndarray]], float]:
    """
    Reads Azure Kinect MKV, runs MediaPipe Hands on color,
    converts landmarks to 3D using transformed depth + calibration,
    and simultaneously writes an overlay video with the 2D skeleton
    and wrist (x,y,z).

    Returns:
        frames_xyz: list over frames;
                    each element is a list of hands;
                    each hand is (21,3) array of smoothed XYZ in meters.
        fps: frames per second used for output videos.
    """
    playback = PyK4APlayback(str(input_video))
    playback.open()

    fps = getattr(playback, "fps", 30.0)
    calib = playback.calibration

    frames_xyz: List[List[np.ndarray]] = []
    overlay_writer = None

    # For temporal smoothing
    prev_smoothed_hands: Optional[List[np.ndarray]] = None
    alpha = 0.4           # EMA weight
    max_jump_m = 0.20     # 20 cm jump treated as outlier

    frame_idx = 0
    while True:
        try:
            cap = playback.get_next_capture()
        except EOFError:
            break
        if cap is None:
            break

        # ---- Depth (aligned to color) ----
        depth = cap.transformed_depth
        if depth is None:
            depth = cap.depth
            if depth is None:
                frames_xyz.append([])
                frame_idx += 1
                continue

        # ---- Color (likely MJPG → decode) ----
        color = cap.color
        if color is None or color.size == 0:
            frames_xyz.append([])
            frame_idx += 1
            continue

        if color.ndim == 1:
            # MJPG encoded; decode to BGR
            color_bgr = cv2.imdecode(color, cv2.IMREAD_COLOR)
            if color_bgr is None:
                frames_xyz.append([])
                frame_idx += 1
                continue
        elif color.ndim == 3:
            if color.shape[2] >= 3:
                color_bgr = color[:, :, :3]
            else:
                gray = color[:, :, 0]
                color_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif color.ndim == 2:
            color_bgr = cv2.cvtColor(color, cv2.COLOR_GRAY2BGR)
        else:
            frames_xyz.append([])
            frame_idx += 1
            continue

        h, w, _ = color_bgr.shape

        # Initialize overlay writer on first valid frame
        if overlay_writer is None:
            output_overlay_video.parent.mkdir(parents=True, exist_ok=True)
            overlay_writer = cv2.VideoWriter(
                str(output_overlay_video),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (w, h),
            )

        # ---- MediaPipe on RGB ----
        frame_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # We'll draw skeleton on a copy in RGB space first
        overlay_rgb = frame_rgb.copy()

        hands_xyz_raw: List[np.ndarray] = []

        if results.multi_hand_landmarks:
            # 1) Draw 2D skeleton using MediaPipe drawing utils
            for lm_set in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    overlay_rgb,
                    lm_set,
                    mp_hands.HAND_CONNECTIONS,
                )

            # 2) Compute raw 3D XYZ per hand
            for lm_set in results.multi_hand_landmarks:
                pts = []
                for i, lm in enumerate(lm_set.landmark):
                    u = int(lm.x * w)
                    v = int(lm.y * h)

                    if 0 <= u < w and 0 <= v < h:
                        d_mm = sample_depth_safely(depth, u, v, radius=1)
                        xyz = to_3d(calib, u, v, d_mm)
                    else:
                        xyz = np.array([np.nan, np.nan, np.nan], dtype=np.float32)

                    pts.append(xyz)

                pts_arr = np.array(pts, dtype=np.float32)  # (21,3)
                hands_xyz_raw.append(pts_arr)
        else:
            hands_xyz_raw = []

        # ---- Temporal smoothing on 3D points ----
        smoothed_hands: List[np.ndarray] = []

        if prev_smoothed_hands is None:
            # First frame with hands → just take raw as smoothed
            for hand in hands_xyz_raw:
                smoothed_hands.append(hand.astype(np.float32))
        else:
            # Smooth up to 2 hands by index
            max_hands = max(len(hands_xyz_raw), len(prev_smoothed_hands))
            for h_idx in range(max_hands):
                if h_idx < len(hands_xyz_raw):
                    raw = hands_xyz_raw[h_idx].astype(np.float32)
                else:
                    # No raw for this index → keep previous if exists
                    if h_idx < len(prev_smoothed_hands):
                        smoothed_hands.append(prev_smoothed_hands[h_idx])
                    continue

                if h_idx < len(prev_smoothed_hands):
                    prev = prev_smoothed_hands[h_idx].astype(np.float32)

                    # Clamp crazy jumps before applying EMA
                    diff = raw - prev
                    dist = np.linalg.norm(diff, axis=1)
                    mask_big = dist > max_jump_m
                    raw_clamped = raw.copy()
                    # For big jumps where previous was valid, keep previous
                    raw_clamped[mask_big & ~np.isnan(prev).any(axis=1)] = prev[
                        mask_big & ~np.isnan(prev).any(axis=1)
                    ]

                    smoothed = alpha * raw_clamped + (1.0 - alpha) * prev
                else:
                    # No previous → no smoothing
                    smoothed = raw

                smoothed_hands.append(smoothed.astype(np.float32))

        prev_smoothed_hands = smoothed_hands if smoothed_hands else prev_smoothed_hands

        # Limit to 2 hands in stored frames for consistency
        frames_xyz.append(smoothed_hands[:2])

        # ---- Draw wrist XYZ text using smoothed values (if available) ----
        if smoothed_hands:
            for hand_idx, hand_pts in enumerate(smoothed_hands[:2]):
                wrist_xyz = hand_pts[0]  # landmark 0
                if not np.isnan(wrist_xyz).any():
                    wx, wy, wz = wrist_xyz
                    # Use wrist 2D location from raw mediapipe (approx: first hand)
                    lm_set = results.multi_hand_landmarks[hand_idx]
                    lm_wrist = lm_set.landmark[0]
                    u = int(lm_wrist.x * w)
                    v = int(lm_wrist.y * h)
                    x_text = max(0, u - 10)
                    y_text = max(20, v - 10)
                    text = f"H{hand_idx} wrist: ({wx:.2f}, {wy:.2f}, {wz:.2f}) m"
                    cv2.putText(
                        overlay_rgb,
                        text,
                        (x_text, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

        # Convert overlay_rgb back to BGR for writing
        overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)

        # Write overlay frame
        if overlay_writer is not None:
            overlay_writer.write(overlay_bgr)

        frame_idx += 1

    playback.close()
    if overlay_writer is not None:
        overlay_writer.release()

    print(f"Saved overlay MP4 → {output_overlay_video}")
    return frames_xyz, fps


# ============================
# ==== SAVE CSV ==============
# ============================

def save_csv(frames_xyz: List[List[np.ndarray]]) -> None:
    """
    Save all 3D landmarks to CSV.
    Each row = one frame.
    Columns: h0_lm0_x, h0_lm0_y, h0_lm0_z, ..., h1_lm20_z
    up to 2 hands, 21 landmarks each.
    """
    rows = []
    for fidx, hands_in_frame in enumerate(frames_xyz):
        row = {"frame": fidx}

        for h in range(2):  # up to 2 hands
            if h < len(hands_in_frame):
                pts = hands_in_frame[h]  # (21,3)
            else:
                pts = np.full((21, 3), np.nan, dtype=np.float32)

            for i in range(21):
                row[f"h{h}_lm{i}_x"] = float(pts[i, 0])
                row[f"h{h}_lm{i}_y"] = float(pts[i, 1])
                row[f"h{h}_lm{i}_z"] = float(pts[i, 2])

        rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved CSV → {output_csv}")


# ============================
# ==== 3D VISUALIZATION ======
# ============================

def compute_axis_limits(frames_xyz: List[List[np.ndarray]]):
    """
    Compute global [xmin, xmax, ymin, ymax, zmin, zmax]
    across all frames/hands, ignoring NaNs.
    """
    all_pts = []
    for hands_in_frame in frames_xyz:
        for hand in hands_in_frame:
            all_pts.append(hand)

    if not all_pts:
        return -0.5, 0.5, -0.5, 0.5, 0.2, 1.5

    all_pts = np.vstack(all_pts)
    mask = ~np.isnan(all_pts).any(axis=1)
    all_pts = all_pts[mask]

    if all_pts.size == 0:
        return -0.5, 0.5, -0.5, 0.5, 0.2, 1.5

    xmin, ymin, zmin = all_pts.min(axis=0)
    xmax, ymax, zmax = all_pts.max(axis=0)

    margin = 0.05
    return (
        xmin - margin,
        xmax + margin,
        ymin - margin,
        ymax + margin,
        zmin - margin,
        zmax + margin,
    )


def render_3d(frames_xyz: List[List[np.ndarray]], fps: float) -> None:
    """
    Render a 3D MP4 of both hands moving through space.
    Uses MediaPipe HAND_CONNECTIONS as the skeleton.
    Axes are labeled (meters).
    """
    output_3d_video.parent.mkdir(parents=True, exist_ok=True)

    H, W = 800, 800
    writer = cv2.VideoWriter(
        str(output_3d_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H),
    )

    xmin, xmax, ymin, ymax, zmin, zmax = compute_axis_limits(frames_xyz)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    canvas = FigureCanvas(fig)

    for frame_idx, hands_in_frame in enumerate(frames_xyz):
        ax.clear()

        ax.set_title("3D Hand Tracking (Azure Kinect + MediaPipe)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")

        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.set_zlim([zmin, zmax])

        # Plot each hand
        for hand in hands_in_frame:
            pts = hand  # (21,3)

            # Scatter valid points
            valid = ~np.isnan(pts).any(axis=1)
            pts_valid = pts[valid]
            if pts_valid.shape[0] > 0:
                ax.scatter(pts_valid[:, 0], pts_valid[:, 1], pts_valid[:, 2], s=25)

            # Draw skeleton segments using HAND_CONNECTIONS
            for a, b in HAND_CONNECTIONS:
                if (
                    0 <= a < pts.shape[0]
                    and 0 <= b < pts.shape[0]
                    and not np.isnan(pts[a]).any()
                    and not np.isnan(pts[b]).any()
                ):
                    ax.plot(
                        [pts[a, 0], pts[b, 0]],
                        [pts[a, 1], pts[b, 1]],
                        [pts[a, 2], pts[b, 2]],
                        linewidth=2,
                    )

        canvas.draw()
        buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        img = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # H, W, 4 (RGBA)

        # Convert RGBA → BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        img_bgr = cv2.resize(img_bgr, (W, H))
        writer.write(img_bgr)

    writer.release()
    plt.close(fig)
    print(f"Saved 3D MP4 → {output_3d_video}")


# ============================
# ========= MAIN =============
# ============================

def main():
    print(f"Opening MKV: {input_video}")
    frames_xyz, fps = extract_3d_landmarks_and_overlay()
    print(f"Total frames processed: {len(frames_xyz)}, fps={fps}")
    save_csv(frames_xyz)
    render_3d(frames_xyz, fps)
    print("Done.")


if __name__ == "__main__":
    main()
