from __future__ import annotations

from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


# ==== USER CONFIGURATION ====
input_video = Path("demo-content/output_with_hands.mp4")   # <-- change this
output_csv = Path("outputs/hand_landmarks_with_pose.csv")  # <-- change this
output_video = Path("outputs/output_debug_video.mp4")      # set to None to skip
max_hands = 2
min_detection_confidence = 0.5
min_tracking_confidence = 0.5

# Choose how to set translation tvec: "centroid" uses mean of all 21 world points; "wrist" uses wrist point
TVEC_MODE = "centroid"   # or "wrist"
# ============================


mp_hands = mp.solutions.hands
_HAND_LANDMARKS = list(mp_hands.HandLandmark)

# Landmark indices for sign disambiguation of PCA axes
WRIST = mp_hands.HandLandmark.WRIST.value
INDEX_MCP = mp_hands.HandLandmark.INDEX_FINGER_MCP.value
PINKY_MCP = mp_hands.HandLandmark.PINKY_MCP.value
MIDDLE_TIP = mp_hands.HandLandmark.MIDDLE_FINGER_TIP.value


def build_columns() -> list[str]:
    cols = ["time_sec", "frame_index", "hand_index", "hand_label", "hand_score"]
    for lm in _HAND_LANDMARKS:
        name = lm.name.lower()
        cols.extend([f"{name}_x", f"{name}_y", f"{name}_z"])
    cols += ["rvec_x", "rvec_y", "rvec_z", "tvec_x", "tvec_y", "tvec_z"]
    return cols


def _np(pt) -> np.ndarray:
    return np.array([pt.x, pt.y, pt.z], dtype=np.float64)


def pca_pose_from_all_world(world_lms) -> tuple[np.ndarray, np.ndarray]:
    P = np.stack([_np(world_lms.landmark[i]) for i in range(21)], axis=0)
    C = P.mean(axis=0)
    X = P - C

    if not np.isfinite(X).all() or np.linalg.norm(X) < 1e-9:
        return np.eye(3, dtype=np.float64), C.reshape(3, 1)

    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt.T
    y_hat = V[:, 0]
    x_hat = V[:, 1]
    z_hat = np.cross(x_hat, y_hat)

    Pw = _np(world_lms.landmark[WRIST])
    Pmt = _np(world_lms.landmark[MIDDLE_TIP])
    Pidx = _np(world_lms.landmark[INDEX_MCP])
    Ppky = _np(world_lms.landmark[PINKY_MCP])

    if np.dot(y_hat, (Pmt - Pw)) < 0:
        y_hat = -y_hat
    if np.dot(x_hat, (Pidx - Ppky)) < 0:
        x_hat = -x_hat

    z_hat = np.cross(x_hat, y_hat)

    def _n(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-12 else v

    x_hat = _n(x_hat)
    y_hat = _n(np.cross(z_hat, x_hat))
    z_hat = _n(np.cross(x_hat, y_hat))

    R = np.column_stack([x_hat, y_hat, z_hat])

    t = Pw.reshape(3, 1) if TVEC_MODE.lower() == "wrist" else C.reshape(3, 1)
    return R, t


def rotation_to_rodrigues(R: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(R)
    return rvec


def main() -> None:
    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_video is not None:
        out_path = output_video.with_suffix(".mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps if fps > 0 else 30.0, (width, height))

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=max_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        model_complexity=1,
    )
    draw = mp.solutions.drawing_utils

    rows: list[list[float]] = []
    frame_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            time_sec = (frame_index - 1) / fps if fps > 0 else 0.0

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                handedness = results.multi_handedness or []
                world_sets = results.multi_hand_world_landmarks or []

                for idx, img_lms in enumerate(results.multi_hand_landmarks):
                    coords: list[float] = []
                    for lm in _HAND_LANDMARKS:
                        p = img_lms.landmark[lm]
                        coords.extend([p.x, p.y, p.z])

                    label = None
                    score = None
                    if idx < len(handedness) and handedness[idx].classification:
                        c = handedness[idx].classification[0]
                        label = c.label.lower()
                        score = c.score

                    rvec_out = [None, None, None]
                    tvec_out = [None, None, None]

                    if idx < len(world_sets):
                        try:
                            R, t = pca_pose_from_all_world(world_sets[idx])
                            rvec = rotation_to_rodrigues(R)
                            rvec_out = [float(rvec[0]), float(rvec[1]), float(rvec[2])]
                            tvec_out = [float(t[0]), float(t[1]), float(t[2])]
                        except Exception:
                            pass

                    rows.append([time_sec, frame_index, idx, label, score] + coords + rvec_out + tvec_out)

                    if writer is not None:
                        draw.draw_landmarks(
                            frame,
                            img_lms,
                            mp_hands.HAND_CONNECTIONS,
                            landmark_drawing_spec=draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            connection_drawing_spec=draw.DrawingSpec(color=(60, 180, 75), thickness=2),
                        )

                        # === Overlay R and T vectors ===
                        if all(v is not None for v in rvec_out + tvec_out):
                            wrist_lm = img_lms.landmark[WRIST]
                            x_px = int(wrist_lm.x * width)
                            y_px = int(wrist_lm.y * height)

                            overlay_text = (
                                f"R: [{rvec_out[0]:.2f}, {rvec_out[1]:.2f}, {rvec_out[2]:.2f}]\n"
                                f"T: [{tvec_out[0]:.2f}, {tvec_out[1]:.2f}, {tvec_out[2]:.2f}]"
                            )

                            for i, line in enumerate(overlay_text.split("\n")):
                                cv2.putText(
                                    frame,
                                    line,
                                    (x_px, y_px + i * 20),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (255, 255, 255),
                                    1,
                                    cv2.LINE_AA,
                                )

            if writer is not None:
                writer.write(frame)
    finally:
        cap.release()
        hands.close()
        if writer is not None:
            writer.release()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        df = pd.DataFrame(rows, columns=build_columns())
        df.to_csv(output_csv, index=False)
        print(f"Processed {frame_index} frames; wrote {len(rows)} detections")
        print(f"Saved CSV: {output_csv.resolve()}")
        if output_video is not None:
            print(f"Preview video: {out_path.resolve()}")
    else:
        df = pd.DataFrame(columns=build_columns())
        df.to_csv(output_csv, index=False)
        print("No hands detected — wrote header-only CSV:", output_csv.resolve())


if __name__ == "__main__":
    main()
