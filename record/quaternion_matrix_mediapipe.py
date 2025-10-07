from __future__ import annotations

from pathlib import Path
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np # <-- NEW: Import NumPy


# ==== USER CONFIGURATION ====
input_video = Path("demo-content/output_with_hands.mp4")
output_csv = Path("outputs/hand_world_landmarks_with_quaternion.csv") # <-- Renamed
output_video = Path("outputs/output_debug_video_with_quaternion.mp4")
max_hands = 2
min_detection_confidence = 0.5
min_tracking_confidence = 0.5
# ============================


_HAND_LANDMARKS = list(mp.solutions.hands.HandLandmark)

# Define landmarks used to create the hand's local coordinate system
# These indices correspond to the HandLandmark enumeration (e.g., WRIST=0)
LM_WRIST = 0
LM_MID_MCP = 9
LM_PINKY_MCP = 17


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a 3D vector."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else np.zeros_like(v)


def rot_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix R to a quaternion [w, x, y, z].
    Source: http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
        
    return np.array([w, x, y, z])


def calculate_hand_quaternion(world_landmarks) -> np.ndarray | None:
    """
    Calculates the hand's quaternion orientation based on 3 world landmarks.
    The local coordinate system is defined as:
    - Origin: WRIST (LM_WRIST)
    - Y-axis (Up palm): Vector from WRIST to MIDDLE_FINGER_MCP (LM_MID_MCP)
    - X-axis (Across palm): Vector from WRIST to PINKY_MCP (LM_PINKY_MCP)
    - Z-axis (Normal): Cross product of Y and X vectors (Y x X)
    """
    try:
        # 1. Get world coordinates for the key points (x, y, z in meters)
        wrist_pos = np.array([
            world_landmarks.landmark[LM_WRIST].x,
            world_landmarks.landmark[LM_WRIST].y,
            world_landmarks.landmark[LM_WRIST].z
        ])
        mid_mcp_pos = np.array([
            world_landmarks.landmark[LM_MID_MCP].x,
            world_landmarks.landmark[LM_MID_MCP].y,
            world_landmarks.landmark[LM_MID_MCP].z
        ])
        pinky_mcp_pos = np.array([
            world_landmarks.landmark[LM_PINKY_MCP].x,
            world_landmarks.landmark[LM_PINKY_MCP].y,
            world_landmarks.landmark[LM_PINKY_MCP].z
        ])

        # 2. Define basis vectors for the local coordinate system
        # Y-axis (Up palm, roughly)
        y_axis = mid_mcp_pos - wrist_pos
        # X-axis (Across palm) - from wrist towards pinky side
        x_axis = pinky_mcp_pos - wrist_pos
        # Z-axis (Normal to palm, right-hand rule: Y x X)
        z_axis = np.cross(y_axis, x_axis)
        
        # 3. Re-orthogonalize and normalize the basis vectors
        # Use Z and Y to re-derive a truly orthogonal X
        z_axis = normalize(z_axis)
        
        # Re-derive Y (Z x X), but using a robust basis (e.g., Y/Middle finger) is better.
        # Let's stick to X, Y, Z for the rotation matrix, using normalized vectors:
        y_axis = normalize(y_axis)
        
        # Calculate X as the cross-product of Y and Z to ensure orthogonality:
        x_axis = np.cross(y_axis, z_axis)
        x_axis = normalize(x_axis) # Should already be unit length if Y and Z were.

        # Correct Order: Z = Cross(Y, X). Let's define the rotation matrix by the normalized axes.
        # X_local -> X_world, Y_local -> Y_world, Z_local -> Z_world
        
        # Rotation Matrix R: columns are the world coordinates of the local X, Y, Z axes.
        R = np.array([x_axis, y_axis, z_axis]).T

        # 4. Convert the Rotation Matrix to a Quaternion [w, x, y, z]
        return rot_matrix_to_quaternion(R)

    except Exception:
        # Handle cases where calculation might fail (e.g., three points are collinear)
        return None


def build_columns() -> list[str]:
    """Builds column names including quaternion and 3D world coordinates."""
    # --- NEW: Add Quaternion columns ---
    columns = ["time_sec", "frame_index", "hand_index", "hand_label", "hand_score"]
    columns.extend(["hand_rot_w", "hand_rot_x", "hand_rot_y", "hand_rot_z"])
    
    # --- Existing: Add 3D World Landmark columns ---
    for landmark in _HAND_LANDMARKS:
        name = landmark.name.lower()
        # Indicate that the coordinates are 3D World Coordinates in meters
        columns.extend([f"{name}_world_x (m)", f"{name}_world_y (m)", f"{name}_world_z (m)"])
    return columns


def main() -> None:
    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=max_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    drawing = mp.solutions.drawing_utils

    rows: list[list[float | str | None]] = []
    frame_index = 0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            frame_index += 1
            time_sec = (frame_index - 1) / fps if fps > 0 else 0.0

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            # Process world landmarks for 3D data and quaternion
            if results.multi_hand_world_landmarks:
                handedness = results.multi_handedness or []
                
                for idx, world_landmarks in enumerate(results.multi_hand_world_landmarks):
                    
                    # --- NEW: Calculate Quaternion Orientation ---
                    quat = calculate_hand_quaternion(world_landmarks)
                    if quat is not None:
                        quat_list = quat.tolist()
                    else:
                        # Use None/NaN placeholders if quaternion calculation fails
                        quat_list = [np.nan, np.nan, np.nan, np.nan] 
                    
                    # --- Existing: Extract Positional Coordinates ---
                    coords: list[float] = []
                    for landmark in _HAND_LANDMARKS:
                        lm = world_landmarks.landmark[landmark]
                        # 'lm' will contain the world coordinates (x, y, z in meters)
                        coords.extend([lm.x, lm.y, lm.z])

                    label = None
                    score = None
                    if idx < len(handedness) and handedness[idx].classification:
                        classification = handedness[idx].classification[0]
                        label = classification.label.lower()
                        score = classification.score

                    # Combine Frame info, Hand info, Quaternion, and Positional Coordinates
                    row_data = [time_sec, frame_index, idx, label, score] + quat_list + coords
                    rows.append(row_data)

                    # --- Use multi_hand_landmarks for drawing (image coordinates) ---
                    if writer is not None and results.multi_hand_landmarks:
                        drawing.draw_landmarks(
                            frame,
                            results.multi_hand_landmarks[idx], # Get the corresponding image landmarks
                            mp.solutions.hands.HAND_CONNECTIONS,
                            landmark_drawing_spec=drawing.DrawingSpec(
                                color=(0, 255, 0), thickness=2, circle_radius=2
                            ),
                            connection_drawing_spec=drawing.DrawingSpec(
                                color=(60, 180, 75), thickness=2
                            ),
                        )
            if writer is not None:
                writer.write(frame)
    finally:
        cap.release()
        hands.close()
        if writer is not None:
            writer.release()

    # Save to CSV
    if rows:
        df = pd.DataFrame(rows, columns=build_columns())
        df.to_csv(output_csv, index=False)
        print(f"Processed {frame_index} frames; stored {len(rows)} hand detections with 3D coordinates and quaternion.")
        print(f"Data written to {output_csv.resolve()}")
    else:
        df = pd.DataFrame(columns=build_columns())
        df.to_csv(output_csv, index=False)
        print("No hand landmarks detected; wrote header-only CSV.")

    if write_video:
        print(f"Preview video written to {out_path.resolve()}")


if __name__ == "__main__":
    main()