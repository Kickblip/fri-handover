import cv2
import mediapipe as mp
import numpy as np

# Input and output file paths
input_video_path = '1_sept21_1_sept21_2.mkv'
output_video_path = 'output_with_hands.mp4'

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Drawing utility
mp_drawing = mp.solutions.drawing_utils

# Read input video
cap = cv2.VideoCapture(input_video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define codec and create VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# -------- NumPy storage (list, then convert to np.array) --------
hand_data = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Current timestamp
    frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    time_sec = frame_index / fps

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe
    results = hands.process(rgb_frame)

    # Save coordinates if hands are found
    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])  # flatten into [x,y,z,...]

            row = [time_sec, hand_idx] + coords
            hand_data.append(row)

            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

    # Write video frame
    out.write(frame)

# Release
cap.release()
out.release()
hands.close()

# Convert to NumPy
hand_data = np.array(hand_data)

print(f"Processing complete. Saved video: {output_video_path}")
print(f"Hand data shape: {hand_data.shape}")  # (N, 65)
print("First row example:\n", hand_data[0])
