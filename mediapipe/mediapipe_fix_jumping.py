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
    "/home/bwilab/Documents/fri-handover/dataset/input_video/1_video.mkv"
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
    h, w = depth.shape
    u0, u1 = max(0, u - radius), min(w, u + radius + 1)
    v0, v1 = max(0, v - radius), min(h, v + radius + 1)

    patch = depth[v0:v1, u0:u1]
    valid = patch[patch > 0]
    if valid.size == 0:
        return 0.0
    return float(np.median(valid))


def to_3d(calib, u: int, v: int, depth_mm: float) -> np.ndarray:
    if depth_mm <= 0:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float32)

    try:
        x_mm, y_mm, z_mm = calib.convert_2d_to_3d(
            (u, v),
            float(depth_mm),
            CalibrationType.COLOR,
            CalibrationType.COLOR,
        )
        return np.array([x_mm, y_mm, z_mm], dtype=np.float32) / 1000.0
    except Exception:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float32)


# ============================
# ==== EXTRACT + OVERLAY =====
# ============================

def extract_3d_landmarks_and_overlay() -> Tuple[List[List[np.ndarray]], float]:
    playback = PyK4APlayback(str(input_video))
    playback.open()

    fps = getattr(playback, "fps", 30.0)
    calib = playback.calibration

    frames_xyz: List[List[np.ndarray]] = []
    overlay_writer = None

    prev_smoothed_hands: Optional[List[np.ndarray]] = None
    alpha = 0.4
    max_jump_m = 0.20

    frame_idx = 0

    while True:
        try:
            cap = playback.get_next_capture()
        except EOFError:
            break
        if cap is None:
            break

        depth = cap.transformed_depth
        if depth is None:
            depth = cap.depth
            if depth is None:
                frames_xyz.append([])
                continue

        color = cap.color
        if color is None or color.size == 0:
            frames_xyz.append([])
            continue

        if color.ndim == 1:
            color_bgr = cv2.imdecode(color, cv2.IMREAD_COLOR)
            if color_bgr is None:
                frames_xyz.append([])
                continue
        elif color.ndim == 3:
            color_bgr = color[:, :, :3] if color.shape[2] >= 3 else cv2.cvtColor(color[:, :, 0], cv2.COLOR_GRAY2BGR)
        else:
            color_bgr = cv2.cvtColor(color, cv2.COLOR_GRAY2BGR)

        h, w, _ = color_bgr.shape

        if overlay_writer is None:
            output_overlay_video.parent.mkdir(parents=True, exist_ok=True)
            overlay_writer = cv2.VideoWriter(
                str(output_overlay_video),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (w, h),
            )

        frame_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        overlay_rgb = frame_rgb.copy()

        hands_xyz_raw: List[np.ndarray] = []

        if results.multi_hand_landmarks:
            for lm_set in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    overlay_rgb,
                    lm_set,
                    mp_hands.HAND_CONNECTIONS,
                )

            for lm_set in results.multi_hand_landmarks:
                pts = []
                for lm in lm_set.landmark:
                    u = int(lm.x * w)
                    v = int(lm.y * h)

                    if 0 <= u < w and 0 <= v < h:
                        d_mm = sample_depth_safely(depth, u, v)
                        xyz = to_3d(calib, u, v, d_mm)
                    else:
                        xyz = np.array([np.nan, np.nan, np.nan], dtype=np.float32)

                    pts.append(xyz)
                hands_xyz_raw.append(np.array(pts, dtype=np.float32))

        # ========== FIXED SMOOTHING LOGIC ==============
        smoothed_hands: List[np.ndarray] = []

        if prev_smoothed_hands is None:
            smoothed_hands = hands_xyz_raw
        else:
            for h_idx in range(len(hands_xyz_raw)):
                raw = hands_xyz_raw[h_idx]

                if h_idx < len(prev_smoothed_hands):
                    prev = prev_smoothed_hands[h_idx]
                    diff = raw - prev
                    dist = np.linalg.norm(diff, axis=1)
                    mask_big = dist > max_jump_m

                    raw_clamped = raw.copy()
                    raw_clamped[mask_big & ~np.isnan(prev).any(axis=1)] = prev[
                        mask_big & ~np.isnan(prev).any(axis=1)
                    ]

                    smoothed = alpha * raw_clamped + (1 - alpha) * prev
                else:
                    smoothed = raw

                smoothed_hands.append(smoothed.astype(np.float32))

        prev_smoothed_hands = smoothed_hands if smoothed_hands else prev_smoothed_hands
        frames_xyz.append(smoothed_hands[:2])

        # ========== FIX: SAFE WRIST-TEXT OVERLAY ==========
        if smoothed_hands and results.multi_hand_landmarks:
            num_text_hands = min(len(smoothed_hands), len(results.multi_hand_landmarks), 2)

            for hand_idx in range(num_text_hands):
                hand_pts = smoothed_hands[hand_idx]
                wrist_xyz = hand_pts[0]

                if not np.isnan(wrist_xyz).any():
                    wx, wy, wz = wrist_xyz

                    lm_set = results.multi_hand_landmarks[hand_idx]
                    lm_wrist = lm_set.landmark[0]

                    u = int(lm_wrist.x * w)
                    v = int(lm_wrist.y * h)

                    cv2.putText(
                        overlay_rgb,
                        f"H{hand_idx} wrist: ({wx:.2f}, {wy:.2f}, {wz:.2f}) m",
                        (max(0, u - 10), max(20, v - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

        overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
        overlay_writer.write(overlay_bgr)

        frame_idx += 1

    playback.close()
    if overlay_writer:
        overlay_writer.release()

    print(f"Saved overlay MP4 → {output_overlay_video}")
    return frames_xyz, fps


# ============================
# ==== SAVE CSV ==============
# ============================

def save_csv(frames_xyz: List[List[np.ndarray]]) -> None:
    rows = []
    for fidx, hands_in_frame in enumerate(frames_xyz):
        row = {"frame": fidx}

        for h in range(2):
            pts = hands_in_frame[h] if h < len(hands_in_frame) else np.full((21, 3), np.nan)

            for i in range(21):
                row[f"h{h}_lm{i}_x"] = float(pts[i, 0])
                row[f"h{h}_lm{i}_y"] = float(pts[i, 1])
                row[f"h{h}_lm{i}_z"] = float(pts[i, 2])

        rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"Saved CSV → {output_csv}")


# ============================
# ==== 3D VISUALIZATION ======
# ============================

def compute_axis_limits(frames_xyz: List[List[np.ndarray]]):
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

    for hands_in_frame in frames_xyz:
        ax.clear()

        ax.set_title("3D Hand Tracking (Azure Kinect + MediaPipe)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")

        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.set_zlim([zmin, zmax])

        for hand in hands_in_frame:
            pts = hand

            valid = ~np.isnan(pts).any(axis=1)
            pts_valid = pts[valid]
            if pts_valid.size > 0:
                ax.scatter(pts_valid[:, 0], pts_valid[:, 1], pts_valid[:, 2], s=25)

            for a, b in HAND_CONNECTIONS:
                if (
                    not np.isnan(pts[a]).any()
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
        img = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
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
