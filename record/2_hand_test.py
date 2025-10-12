from __future__ import annotations

from pathlib import Path
import os
import csv
from typing import List, Dict, Any, Optional

import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from pyk4a import PyK4APlayback, CalibrationType

# ============================
# ==== USER CONFIGURATION ====
# ============================
# Input/Output Paths
# input_video = Path("1_bhavana_final_neel_final.mkv")      # <-- Change this to your video path
input_video = Path("record/trials/2/6_rohan_diego.mkv")
output_csv = Path("outputs/2_handover_cartesian_combined.csv")          # one row per frame
output_video = Path("outputs/2_output_world_coordinates_preview.mp4")   # optional preview video
max_hands = 2  # allows detection of two hands
min_detection_confidence = 0.5
min_tracking_confidence = 0.5
# ============================

mp_hands = mp.solutions.hands

# MediaPipe Hand Landmarks
_HAND_LANDMARKS = list(mp_hands.HandLandmark)

# ---------- geometry helpers (2D) ----------
def point_to_segment_dist2(px, py, x1, y1, x2, y2):
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1
    L2 = vx*vx + vy*vy
    if L2 == 0:
        dx, dy = px - x1, py - y1
        return dx*dx + dy*dy
    t = (wx*vx + wy*vy) / L2
    if t < 0.0: t = 0.0
    elif t > 1.0: t = 1.0
    projx, projy = x1 + t*vx, y1 + t*vy
    dx, dy = px - projx, py - projy
    return dx*dx + dy*dy

def min_dist2_point_to_poly(px, py, poly):
    best = float('inf')
    m = len(poly)
    for i in range(m):
        x1, y1 = poly[i]
        x2, y2 = poly[(i+1) % m]
        d2 = point_to_segment_dist2(px, py, x1, y1, x2, y2)
        if d2 < best:
            best = d2
    return best

# ---------- Kinect intrinsics from the MKV ----------
def load_k4a_calib(mkv_path: str):
    pb = PyK4APlayback(str(mkv_path))
    pb.open()
    K = pb.calibration.get_camera_matrix(CalibrationType.COLOR).astype(np.float32)
    dist = pb.calibration.get_distortion_coefficients(CalibrationType.COLOR).astype(np.float32)
    pb.close()
    return K, dist

# ---------- Read AprilTag *_vertices.csv into dict: frame_idx -> (8x3) ----------
def load_vertices_csv(csv_path: str):
    table: Dict[int, np.ndarray] = {}
    if not os.path.exists(csv_path):
        return table
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fi = int(row["frame_idx"])
            # skip no-detect rows (tag_id = -1)
            try:
                if int(float(row["tag_id"])) < 0:
                    continue
            except Exception:
                continue
            V = np.zeros((8, 3), np.float32)
            ok = True
            for i in range(8):
                try:
                    V[i, 0] = float(row[f"v{i}_x"])
                    V[i, 1] = float(row[f"v{i}_y"])
                    V[i, 2] = float(row[f"v{i}_z"])
                except Exception:
                    ok = False
                    break
            if ok:
                table[fi] = V
    return table

# ---------- Project 8x3 camera-frame verts to image pixels ----------
_Z3 = np.zeros((3,1), np.float32)
def verts_cam_to_pixels(verts_cam_8x3: np.ndarray, K: np.ndarray, dist: np.ndarray):
    pts2d, _ = cv2.projectPoints(verts_cam_8x3.astype(np.float32), _Z3, _Z3, K, dist)
    pts2d = np.round(pts2d[:, 0]).astype(int)  # shape (8,2)
    poly = [(int(x), int(y)) for (x, y) in pts2d]
    # robust pixel "center": mean of projected vertices
    cx = int(np.mean(pts2d[:, 0])); cy = int(np.mean(pts2d[:, 1]))
    return (cx, cy), poly

def get_landmark_feature_names() -> List[str]:
    """
    Generates the list of 63 base coordinate feature names without hand suffixes.
    Example: ['wrist_world_x', 'wrist_world_y', 'wrist_world_z', ...]
    """
    columns = []
    for landmark in _HAND_LANDMARKS:
        name = landmark.name.lower()
        columns.extend([f"{name}_world_x", f"{name}_world_y", f"{name}_world_z"])
    return columns

def get_final_columns() -> List[str]:
    """
    Generates the full list of columns for the final, pivoted CSV.
    """
    base_cols = ["time_sec", "frame_index"]
    # Metadata for each hand
    meta_cols = [f"hand_label_{i}" for i in range(max_hands)] + \
                [f"hand_score_{i}" for i in range(max_hands)]
    # Coordinate columns for each hand (126 columns total for 2 hands)
    feature_names = get_landmark_feature_names()
    coord_cols = []
    for i in range(max_hands):
        coord_cols.extend([f"{name}_{i}" for name in feature_names])
    # NEW: box pixels + holding flags
    extra = ["box_center_x_px", "box_center_y_px", "hand_holding_0", "hand_holding_1"]
    return base_cols + meta_cols + coord_cols + extra

def extract_coords_and_meta(results: Any, idx: int) -> tuple[List[float], Optional[str], Optional[float]]:
    """Helper to extract coordinates, label, and score for a single hand."""
    coords: List[float] = []
    # Extract World Coordinates (meters)
    if results.multi_hand_world_landmarks:
        world_landmarks = results.multi_hand_world_landmarks[idx]
        for landmark in _HAND_LANDMARKS:
            lm = world_landmarks.landmark[landmark]
            coords.extend([lm.x, lm.y, lm.z])
    # Extract Handedness (Label and Score)
    label = None
    score = None
    if results.multi_handedness and idx < len(results.multi_handedness):
        handedness = results.multi_handedness[idx]
        if handedness.classification:
            classification = handedness.classification[0]
            label = classification.label.lower()
            score = classification.score
    return coords, label, score

def main() -> None:
    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Video Writer setup
    write_video = output_video is not None
    if write_video:
        out_path = output_video.with_suffix(".mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(out_path),
            fourcc,
            fps if fps > 0 else 30.0,
            (width, height),
        )
    else:
        writer = None

    # MediaPipe Hands setup
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=max_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    drawing = mp.solutions.drawing_utils

    # Load Kinect intrinsics (same MKV) and the AprilTag vertices CSV
    K, dist = load_k4a_calib(str(input_video))
    root = str(input_video.with_suffix(""))  # drop extension
    vertices_csv_path = root + "_vertices.csv"
    verts_by_frame = load_vertices_csv(vertices_csv_path)

    rows: List[Dict[str, Any]] = []
    frame_index = 0
    feature_names = get_landmark_feature_names()
    final_columns = get_final_columns()

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_index += 1  # NOTE: AprilTag CSV uses 0-based; we’ll query with (frame_index - 1)
            time_sec = (frame_index - 1) / fps if fps > 0 else 0.0

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            # Initialize a dictionary for the single output row for this frame
            row_data: Dict[str, Any] = {"time_sec": time_sec, "frame_index": frame_index}

            # ---- Select interacting hands using AprilTag polygon in pixel space ----
            # Lookup AprilTag 3D verts for this 0-based frame index
            V = verts_by_frame.get(frame_index - 1, None)
            box_center_px, box_poly_px = None, None
            if V is not None:
                (cx, cy), poly = verts_cam_to_pixels(V, K, dist)
                box_center_px, box_poly_px = (cx, cy), poly

            row_data_box_cx = np.nan if box_center_px is None else box_center_px[0]
            row_data_box_cy = np.nan if box_center_px is None else box_center_px[1]

            selected_indices: List[int] = []
            hand_entries: List[Dict[str, Any]] = []

            # Collect pixel wrists + fingertips for scoring
            if results.multi_hand_landmarks:
                for i, hand_lms in enumerate(results.multi_hand_landmarks):
                    # wrist in pixels
                    wx = int(hand_lms.landmark[mp_hands.HandLandmark.WRIST].x * width)
                    wy = int(hand_lms.landmark[mp_hands.HandLandmark.WRIST].y * height)
                    # fingertips (thumb/index/middle/ring/pinky tips)
                    tips = []
                    for t in [4, 8, 12, 16, 20]:
                        tx = int(hand_lms.landmark[t].x * width)
                        ty = int(hand_lms.landmark[t].y * height)
                        tips.append((tx, ty))
                    hand_entries.append({"idx": i, "wrist": (wx, wy), "tips": tips})

            # If we have a box polygon and at least one hand, choose 1 per side by fingertip->polygon distance
            if box_poly_px is not None and hand_entries:
                cx, cy = box_center_px
                scored = []
                for h in hand_entries:
                    d2_min = min(min_dist2_point_to_poly(tx, ty, box_poly_px) for (tx, ty) in h["tips"])
                    side = 'L' if h["wrist"][0] < cx else 'R'
                    scored.append((d2_min, side, h["idx"], h))

                # best left, best right
                left  = min((s for s in scored if s[1] == 'L'), default=None, key=lambda x: x[0])
                right = min((s for s in scored if s[1] == 'R'), default=None, key=lambda x: x[0])
                if left:  selected_indices.append(left[2])
                if right: selected_indices.append(right[2])

                # Fallback: only one side visible → pick a second far enough from the first wrist
                if len(selected_indices) == 1 and len(hand_entries) >= 2:
                    SEP_WRIST2 = 200 * 200  # tune 160–240 depending on FOV/resolution
                    chosen = next(h for (_, _, i, h) in scored if i == selected_indices[0])
                    cw, ch = chosen["wrist"]
                    best_i, best_val = None, float('inf')
                    # prefer hands close to the box (smaller d2_min first)
                    for (d2, _, i, h) in sorted(scored, key=lambda x: x[0]):
                        if i == selected_indices[0]:
                            continue
                        dx, dy = h["wrist"][0] - cw, h["wrist"][1] - ch
                        dd = dx * dx + dy * dy
                        if dd >= SEP_WRIST2 and dd < best_val:
                            best_val, best_i = dd, i
                    if best_i is not None:
                        selected_indices.append(best_i)

            # ---- Populate metadata/coords for expected hands (0 and 1) ----
            detected_indices = set()
            num_detected = len(results.multi_hand_world_landmarks) if results.multi_hand_world_landmarks else 0

            if num_detected > 0:
                for idx in range(num_detected):
                    coords, label, score = extract_coords_and_meta(results, idx)
                    if idx < max_hands:
                        # metadata
                        row_data[f"hand_label_{idx}"] = label
                        row_data[f"hand_score_{idx}"] = score
                        # world coords
                        for feature_name, coord_value in zip(get_landmark_feature_names(), coords):
                            row_data[f"{feature_name}_{idx}"] = coord_value
                        detected_indices.add(idx)

            # Fill missing hand slots with NaN
            for i in range(max_hands):
                if i not in detected_indices:
                    row_data[f"hand_label_{i}"] = np.nan
                    row_data[f"hand_score_{i}"] = np.nan
                    for feature_name in get_landmark_feature_names():
                        row_data[f"{feature_name}_{i}"] = np.nan

            # Add box center + holding flags
            holding_0 = (0 in selected_indices)
            holding_1 = (1 in selected_indices)
            row_data["box_center_x_px"] = row_data_box_cx
            row_data["box_center_y_px"] = row_data_box_cy
            row_data["hand_holding_0"] = bool(holding_0)
            row_data["hand_holding_1"] = bool(holding_1)

            rows.append(row_data)

            # Drawing (image coordinates)
            if writer is not None:
                # draw hands
                if results.multi_hand_landmarks:
                    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            landmark_drawing_spec=drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            connection_drawing_spec=drawing.DrawingSpec(color=(60, 180, 75), thickness=2),
                        )
                # overlay box + selections
                if box_poly_px is not None:
                    cv2.circle(frame, (int(row_data_box_cx), int(row_data_box_cy)), 6, (0, 255, 255), 2)
                    cv2.polylines(frame, [np.array(box_poly_px, np.int32)], True, (0, 255, 255), 2)
                # mark selected hands
                for i, h in enumerate(hand_entries):
                    col = (0, 255, 0) if i in selected_indices else (120, 120, 120)
                    cv2.circle(frame, h["wrist"], 10, col, 2)
                    if i in selected_indices:
                        cv2.putText(frame, "HOLDING", (h["wrist"][0] + 8, h["wrist"][1] - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

                writer.write(frame)

    finally:
        cap.release()
        hands.close()
        if writer is not None:
            writer.release()

    # Save to CSV
    if rows:
        df = pd.DataFrame(rows, columns=final_columns)
        print("NOTE: Coordinates are world coordinates in meters (m).")
        df.to_csv(output_csv, index=False)
        print(f"\n✅ Processing complete. {frame_index} total frames analyzed.")
        print(f"✅ Stored {len(rows)} frames of combined two-hand data.")
        print(f"✅ Combined world Cartesian data written to {output_csv.resolve()}")
    else:
        df = pd.DataFrame(columns=final_columns)
        df.to_csv(output_csv, index=False)
        print("No hand landmarks detected; wrote header-only CSV.")

    if output_video is not None:
        print(f"Preview video written to {output_video.with_suffix('.mp4').resolve()}")

if __name__ == "__main__":
    main()
