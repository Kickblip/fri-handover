from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# Azure Kinect
from pyk4a import (
    PyK4A,
    Config,
    ColorResolution,
    DepthMode,
    FPS,
    CalibrationType,
    PyK4APlayback,
)

# Matplotlib 3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


# ============================
# ==== USER CONFIGURATION ====
# ============================

input_video = Path(
    "/home/bwilab/Documents/fri-handover/dataset/mediapipe_outputs/video/world/1_w_b_world.mp4"
)
output_csv = Path("dataset/mediapipe_outputs/csv/depth_results.csv")
output_video = Path("dataset/mediapipe_outputs/video/3d_visualization_output.mp4")

max_hands = 2
min_detection_confidence = 0.5
min_tracking_confidence = 0.5

VIZ_WIDTH, VIZ_HEIGHT = 1000, 1000
VIZ_SCALE_M = 0.75  # +/- 0.75m in X,Y

# ============================

mp_hands = mp.solutions.hands
_HAND_LANDMARKS = list(mp_hands.HandLandmark)


# ====================================================
# ============ HELPER FUNCTION DEFINITIONS ============
# ====================================================

def get_landmark_feature_names() -> List[str]:
    cols: List[str] = []
    for lm in _HAND_LANDMARKS:
        name = lm.name.lower()
        cols.extend(
            [
                f"{name}_norm_x",
                f"{name}_norm_y",
                f"{name}_px_x",
                f"{name}_px_y",
                f"{name}_depth_m",
                f"{name}_camX",
                f"{name}_camY",
                f"{name}_camZ",
            ]
        )
    return cols


def get_final_columns() -> List[str]:
    base_cols = ["time_sec", "frame_index"]
    meta_cols = [f"hand_label_{i}" for i in range(max_hands)] + [
        f"hand_score_{i}" for i in range(max_hands)
    ]
    feats = get_landmark_feature_names()
    coord_cols: List[str] = []
    for i in range(max_hands):
        coord_cols.extend([f"{name}_{i}" for name in feats])
    return base_cols + meta_cols + coord_cols


def extract_landmarks_with_depth(
    results,
    depth_frame: Optional[np.ndarray],
    intrinsics: np.ndarray,
    width: int,
    height: int,
) -> List[List[float]]:
    """
    Extract normalized (x, y), pixel (x, y), depth, and 3D camera-space coordinates
    for each detected hand.
    """
    if results is None or not results.multi_hand_landmarks:
        return []

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    all_hands: List[List[float]] = []

    for hand_landmarks in results.multi_hand_landmarks:
        coords: List[float] = []
        for lm in _HAND_LANDMARKS:
            lm_obj = hand_landmarks.landmark[lm]

            x_norm = lm_obj.x
            y_norm = lm_obj.y

            x_px = int(np.clip(int(round(x_norm * width)), 0, width - 1))
            y_px = int(np.clip(int(round(y_norm * height)), 0, height - 1))

            if depth_frame is not None:
                raw = depth_frame[y_px, x_px]
                depth_m = raw / 1000.0 if raw > 0 else np.nan
            else:
                depth_m = np.nan

            if not np.isnan(depth_m) and depth_m > 0:
                X = (x_px - cx) * depth_m / fx
                Y = (y_px - cy) * depth_m / fy
                Z = depth_m
            else:
                X = Y = Z = np.nan

            coords.extend([x_norm, y_norm, x_px, y_px, depth_m, X, Y, Z])

        all_hands.append(coords)

    return all_hands


def get_3d_points_from_data(all_hand_data: List[List[float]]) -> List[np.ndarray]:
    frame_3d_points: List[np.ndarray] = []
    for hand in all_hand_data:
        if not hand:
            frame_3d_points.append(np.empty((3, 0)))
            continue

        mat = np.array(hand, dtype=np.float32).reshape(-1, 8)  # (21,8)
        xyz = mat[:, 5:8].T  # (3,21)

        valid = ~np.isnan(xyz).any(axis=0)
        valid_points = xyz[:, valid]  # (3, N_valid)

        frame_3d_points.append(valid_points)

    return frame_3d_points


def plot_3d_landmarks(
    points_list: List[np.ndarray],
    fig_size=(10.0, 10.0),
    scale_m: float = 0.75,
) -> np.ndarray:
    """
    Generate a 3D visualization frame with axes, ticks, and grid.
    Returns: BGR uint8 image for OpenCV.
    """
    fig = plt.figure(figsize=fig_size, dpi=100)  # 1000x1000 px
    ax = fig.add_subplot(111, projection="3d")

    ax.set_title("3D Hand Landmarks (Camera Space)", fontsize=16)
    ax.set_xlabel("X (m)", fontsize=14, labelpad=12)
    ax.set_ylabel("Y (m)", fontsize=14, labelpad=12)
    ax.set_zlabel("Z (m)", fontsize=14, labelpad=12)

    ax.set_facecolor("white")
    ax.xaxis.pane.set_color((1, 1, 1, 1))
    ax.yaxis.pane.set_color((1, 1, 1, 1))
    ax.zaxis.pane.set_color((1, 1, 1, 1))

    ax.tick_params(axis="x", colors="black", labelsize=10)
    ax.tick_params(axis="y", colors="black", labelsize=10)
    ax.tick_params(axis="z", colors="black", labelsize=10)

    ax.set_xlim(-scale_m, scale_m)
    ax.set_ylim(-scale_m, scale_m)
    ax.set_zlim(0.3, scale_m * 2)

    ticks_xy = np.arange(-scale_m, scale_m + 0.001, 0.25)
    ticks_z = np.arange(0.3, scale_m * 2 + 0.001, 0.25)
    ax.set_xticks(ticks_xy)
    ax.set_yticks(ticks_xy)
    ax.set_zticks(ticks_z)

    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.5)
    ax.set_box_aspect([1, 1, 1])  # Equal aspect

    colors = ["#00FF00", "#FF00FF"]
    connections = list(mp_hands.HAND_CONNECTIONS)

    for hand_idx, pts in enumerate(points_list):
        if pts.size == 0:
            continue

        color = colors[hand_idx % len(colors)]
        X, Y, Z = pts[0, :], pts[1, :], pts[2, :]

        ax.scatter(X, Y, Z, c=color, s=40)

        if pts.shape[1] == 21:
            for c in connections:
                i1, i2 = int(c[0]), int(c[1])
                if i1 < 21 and i2 < 21:
                    ax.plot(
                        [pts[0, i1], pts[0, i2]],
                        [pts[1, i1], pts[1, i2]],
                        [pts[2, i1], pts[2, i2]],
                        color=color,
                        linewidth=2,
                    )

    canvas = FigureCanvas(fig)
    canvas.draw()
    w, h = canvas.get_width_height()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    img_rgba = buf.reshape((h, w, 4))
    img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    return img_bgr


# ====================================================
# ============ MODE 1: LIVE KINECT + MP4 =============
# ====================================================

def process_with_live_kinect() -> None:
    print(f"Using OpenCV video + LIVE Azure Kinect depth for: {input_video}")

    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Initializing Azure Kinect (live)...")
    k4a = PyK4A(
        Config(
            color_resolution=ColorResolution.RES_1080P,
            depth_mode=DepthMode.NFOV_UNBINNED,
            synchronized_images_only=True,
        )
    )
    k4a.start()
    intrinsics = k4a.calibration.get_camera_matrix(CalibrationType.COLOR)
    print("Azure Kinect started.")

    # Video writer
    if output_video is not None:
        output_video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (VIZ_WIDTH, VIZ_HEIGHT),
        )
        print(f"3D video output → {output_video}")
    else:
        writer = None

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=max_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    rows: List[Dict[str, Any]] = []
    final_columns = get_final_columns()
    feature_names = get_landmark_feature_names()
    frame_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            time_sec = (frame_index - 1) / fps if fps > 0 else 0.0

            capture = k4a.get_capture()
            depth_frame = capture.transformed_depth
            depth_np = np.asarray(depth_frame) if depth_frame is not None else None

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            row: Dict[str, Any] = {
                "time_sec": time_sec,
                "frame_index": frame_index,
            }
            detected_indices = set()
            frame_3d_points: List[np.ndarray] = []

            if results and results.multi_hand_landmarks:
                all_hand_data = extract_landmarks_with_depth(
                    results, depth_np, intrinsics, width, height
                )
                frame_3d_points = get_3d_points_from_data(all_hand_data)

                for idx in range(min(len(all_hand_data), max_hands)):
                    if results.multi_handedness:
                        row[f"hand_label_{idx}"] = (
                            results.multi_handedness[idx]
                            .classification[0]
                            .label.lower()
                        )
                        row[f"hand_score_{idx}"] = (
                            results.multi_handedness[idx].classification[0].score
                        )
                    else:
                        row[f"hand_label_{idx}"] = np.nan
                        row[f"hand_score_{idx}"] = np.nan

                    hand_flat = all_hand_data[idx]
                    for feat_name, value in zip(feature_names, hand_flat):
                        row[f"{feat_name}_{idx}"] = value

                    detected_indices.add(idx)

            for i in range(max_hands):
                if i not in detected_indices:
                    row[f"hand_label_{i}"] = np.nan
                    row[f"hand_score_{i}"] = np.nan
                    for feat_name in feature_names:
                        row[f"{feat_name}_{i}"] = np.nan

            rows.append(row)

            if writer is not None:
                viz_frame = plot_3d_landmarks(
                    frame_3d_points if frame_3d_points else [],
                    fig_size=(VIZ_WIDTH / 100.0, VIZ_HEIGHT / 100.0),
                    scale_m=VIZ_SCALE_M,
                )
                if viz_frame.shape[1] != VIZ_WIDTH or viz_frame.shape[0] != VIZ_HEIGHT:
                    viz_frame = cv2.resize(viz_frame, (VIZ_WIDTH, VIZ_HEIGHT))
                writer.write(viz_frame)

    finally:
        cap.release()
        k4a.stop()
        hands.close()
        if writer is not None:
            writer.release()
        print("Cleanup complete (live mode).")

    if rows:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows, columns=final_columns)
        df.to_csv(output_csv, index=False)
        print(f"Saved CSV with {len(rows)} frames → {output_csv}")
    else:
        print("No rows collected; CSV not written.")


# ====================================================
# ======== MODE 2: AZURE MKV PLAYBACK (OPTIONAL) =====
# ====================================================

def process_with_playback() -> None:
    print(f"Opening Azure Kinect recording (MKV): {input_video}")
    playback = PyK4APlayback(str(input_video))
    playback.open()
    print("Playback opened.")

    intrinsics = playback.calibration.get_camera_matrix(CalibrationType.COLOR)
    cfg = playback.configuration
    fps_enum = cfg.camera_fps
    if fps_enum == FPS.FPS_5:
        fps = 5.0
    elif fps_enum == FPS.FPS_15:
        fps = 15.0
    elif fps_enum == FPS.FPS_30:
        fps = 30.0
    else:
        fps = 30.0

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=max_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    if output_video is not None:
        output_video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (VIZ_WIDTH, VIZ_HEIGHT),
        )
        print(f"3D video output → {output_video}")
    else:
        writer = None

    rows: List[Dict[str, Any]] = []
    final_columns = get_final_columns()
    feature_names = get_landmark_feature_names()
    frame_index = 0

    try:
        while True:
            try:
                capture = playback.get_next_capture()
            except EOFError:
                print("End of recording reached.")
                break

            if capture is None:
                break

            color = capture.color  # BGRA
            depth = capture.transformed_depth  # mm, aligned

            if color is None or depth is None:
                continue

            frame_index += 1
            time_sec = (frame_index - 1) / fps if fps > 0 else 0.0

            color_rgb = cv2.cvtColor(color, cv2.COLOR_BGRA2RGB)
            h, w = color_rgb.shape[:2]
            depth_np = np.asarray(depth)

            results = hands.process(color_rgb)

            row: Dict[str, Any] = {
                "time_sec": time_sec,
                "frame_index": frame_index,
            }
            detected_indices = set()
            frame_3d_points: List[np.ndarray] = []

            if results and results.multi_hand_landmarks:
                all_hand_data = extract_landmarks_with_depth(
                    results, depth_np, intrinsics, w, h
                )
                frame_3d_points = get_3d_points_from_data(all_hand_data)

                for idx in range(min(len(all_hand_data), max_hands)):
                    if results.multi_handedness:
                        row[f"hand_label_{idx}"] = (
                            results.multi_handedness[idx]
                            .classification[0]
                            .label.lower()
                        )
                        row[f"hand_score_{idx}"] = (
                            results.multi_handedness[idx].classification[0].score
                        )
                    else:
                        row[f"hand_label_{idx}"] = np.nan
                        row[f"hand_score_{idx}"] = np.nan

                    hand_flat = all_hand_data[idx]
                    for feat_name, value in zip(feature_names, hand_flat):
                        row[f"{feat_name}_{idx}"] = value

                    detected_indices.add(idx)

            for i in range(max_hands):
                if i not in detected_indices:
                    row[f"hand_label_{i}"] = np.nan
                    row[f"hand_score_{i}"] = np.nan
                    for feat_name in feature_names:
                        row[f"{feat_name}_{i}"] = np.nan

            rows.append(row)

            if writer is not None:
                viz_frame = plot_3d_landmarks(
                    frame_3d_points if frame_3d_points else [],
                    fig_size=(VIZ_WIDTH / 100.0, VIZ_HEIGHT / 100.0),
                    scale_m=VIZ_SCALE_M,
                )
                if viz_frame.shape[1] != VIZ_WIDTH or viz_frame.shape[0] != VIZ_HEIGHT:
                    viz_frame = cv2.resize(viz_frame, (VIZ_WIDTH, VIZ_HEIGHT))
                writer.write(viz_frame)

    finally:
        playback.close()
        hands.close()
        if writer is not None:
            writer.release()
        print("Cleanup complete (playback mode).")

    if rows:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows, columns=final_columns)
        df.to_csv(output_csv, index=False)
        print(f"Saved CSV with {len(rows)} frames → {output_csv}")
    else:
        print("No rows collected; CSV not written.")


# ====================================================
# ======================== MAIN ======================
# ====================================================

def main() -> None:
    # If it's an Azure MKV recording, use playback. Otherwise, use live depth.
    if input_video.suffix.lower() == ".mkv":
        process_with_playback()
    else:
        process_with_live_kinect()


if __name__ == "__main__":
    main()



