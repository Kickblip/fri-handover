#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from pupil_apriltags import Detector

# ==== USER CONFIGURATION ====
input_video = Path("demo-content/output_with_hands.mp4")       # <-- Change this
output_csv = Path("outputs/hands_box_aware_world.csv")         # <-- Change this
output_video = Path("outputs/hands_box_aware_overlay.mp4")     # <-- Optional, set to None to skip
max_hands = 2                                                  # keep 2 hands: those closest to the box
min_detection_confidence = 0.5
min_tracking_confidence = 0.5
model_complexity = 1

# Box tags (AprilTag IDs) & family
BOX_TAG_IDS = {0, 1}
TAG_FAMILY = "tag36h11"
# ============================

mp_hands = mp.solutions.hands
_HAND_LANDMARKS = list(mp_hands.HandLandmark)
LANDMARK_NAMES = [lm.name.lower() for lm in _HAND_LANDMARKS]


def build_columns() -> List[str]:
    cols: List[str] = ["time_sec", "frame_index"]
    for h in range(max_hands):
        cols += [f"hand_label_{h}", f"hand_score_{h}"]
        for name in LANDMARK_NAMES:
            cols += [f"{name}_world_x_{h}", f"{name}_world_y_{h}", f"{name}_world_z_{h}"]
    return cols


def palm_center_px(landmarks_img, img_w: int, img_h: int) -> Tuple[float, float]:
    """
    Estimate palm center from wrist + MCPs in *image* (normalized) coords.
    """
    LM = mp_hands.HandLandmark
    sel = [LM.WRIST, LM.INDEX_FINGER_MCP, LM.MIDDLE_FINGER_MCP, LM.RING_FINGER_MCP, LM.PINKY_MCP]
    xs, ys = [], []
    for lid in sel:
        lm = landmarks_img.landmark[lid]
        xs.append(lm.x * img_w)
        ys.append(lm.y * img_h)
    return float(np.mean(xs)), float(np.mean(ys))


def extract_world_vec(landmarks_world) -> np.ndarray:
    """Return (21, 3) array of world coords (meters)."""
    out = np.zeros((21, 3), dtype=np.float32)
    for i in range(21):
        lm = landmarks_world.landmark[i]
        out[i, 0] = lm.x
        out[i, 1] = lm.y
        out[i, 2] = lm.z
    return out


def detect_box_center_px(gray: np.ndarray, tag_detector: Detector, valid_ids: set[int]) -> Optional[Tuple[float, float]]:
    dets = tag_detector.detect(gray, estimate_tag_pose=False)
    centers = [d.center for d in dets if d.tag_id in valid_ids]
    if not centers:
        return None
    cx = float(np.mean([c[0] for c in centers]))
    cy = float(np.mean([c[1] for c in centers]))
    return (cx, cy)


def main():
    cols = build_columns()
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Optional writer
    writer = None
    if output_video is not None:
        output_video.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video), fourcc, fps, (W, H))

    # AprilTag detector
    tag_detector = Detector(families=TAG_FAMILY, nthreads=1, quad_decimate=1.0, quad_sigma=0.0)

    # MediaPipe Hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=max(2, max_hands),
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    rows: List[List[Any]] = []
    frame_index = 0
    last_box_center: Optional[Tuple[float, float]] = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            time_sec = frame_index / fps

            # AprilTag pixel center(s)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            box_center = detect_box_center_px(gray, tag_detector, BOX_TAG_IDS)
            if box_center is None:
                box_center = last_box_center
            else:
                last_box_center = box_center

            # MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            row: Dict[str, Any] = {"time_sec": time_sec, "frame_index": frame_index}
            used_slots = set()

            img_lms = results.multi_hand_landmarks
            world_lms = results.multi_hand_world_landmarks
            handedness = results.multi_handedness

            ordering: List[int] = []
            if world_lms:
                n = len(world_lms)
                idxs = list(range(n))
                if box_center is not None and img_lms:
                    pairs = []
                    for i in idxs:
                        pcx, pcy = palm_center_px(img_lms[i], W, H)
                        dist2 = (pcx - box_center[0]) ** 2 + (pcy - box_center[1]) ** 2
                        pairs.append((i, dist2))
                    pairs.sort(key=lambda t: t[1])
                    ordering = [p[0] for p in pairs][:max_hands]
                else:
                    ordering = idxs[:max_hands]

                # Fill slots 0..max_hands-1 with selected hands
                for out_slot, orig_idx in enumerate(ordering):
                    label = None
                    score = None
                    if handedness and len(handedness) > orig_idx and handedness[orig_idx].classification:
                        cat = handedness[orig_idx].classification[0]
                        label = cat.label  # "Left"/"Right"
                        score = float(cat.score)

                    row[f"hand_label_{out_slot}"] = label
                    row[f"hand_score_{out_slot}"] = score

                    world_vec = extract_world_vec(world_lms[orig_idx])  # (21,3)
                    for lid, name in enumerate(LANDMARK_NAMES):
                        row[f"{name}_world_x_{out_slot}"] = float(world_vec[lid, 0])
                        row[f"{name}_world_y_{out_slot}"] = float(world_vec[lid, 1])
                        row[f"{name}_world_z_{out_slot}"] = float(world_vec[lid, 2])

                    used_slots.add(out_slot)

            # Pad missing slots for consistent columns
            for s in range(max_hands):
                if s not in used_slots:
                    row[f"hand_label_{s}"] = None
                    row[f"hand_score_{s}"] = None
                    for name in LANDMARK_NAMES:
                        row[f"{name}_world_x_{s}"] = np.nan
                        row[f"{name}_world_y_{s}"] = np.nan
                        row[f"{name}_world_z_{s}"] = np.nan

            rows.append([row.get(c, np.nan) for c in cols])

            # Debug overlay
            if writer is not None:
                if box_center is not None:
                    cv2.circle(frame, (int(box_center[0]), int(box_center[1])), 8, (0, 255, 0), -1)
                    cv2.putText(frame, "BOX", (int(box_center[0]) + 8, int(box_center[1]) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                if img_lms:
                    chosen = set(ordering[:max_hands]) if ordering else set()
                    for i, lm_img in enumerate(img_lms):
                        pcx, pcy = palm_center_px(lm_img, W, H)
                        color = (200, 200, 200)
                        if i in chosen:
                            if ordering and i == ordering[0]:
                                color = (255, 0, 0)   # slot 0
                            elif len(ordering) > 1 and i == ordering[1]:
                                color = (0, 0, 255)   # slot 1
                        cv2.circle(frame, (int(pcx), int(pcy)), 6, color, -1)
                writer.write(frame)

            frame_index += 1

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        hands.close()

    # Write CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(output_csv, index=False)
    print(f"Wrote CSV: {output_csv}")
    if output_video is not None:
        print(f"Wrote debug video: {output_video}")


if __name__ == "__main__":
    main()
