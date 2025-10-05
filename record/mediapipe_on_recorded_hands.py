import cv2
import mediapipe as mp
import os

# Input and output file paths
input_video_path = '1_bhavana_final_neel_final.mkv
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
all_frame_landmarks = []
# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define codec and create VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for .avi
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe
    results = hands.process(rgb_frame)
2
    # Draw landmarks if any hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Write frame to output video
    out.write(frame)

# Release resources
cap.release()
out.release()
hands.close()

print(f"Processing complete. Saved to: {output_video_path}")




# make sure to include time stamp, list of tuples (each tuple, 21 landmarks and the time stamp)
def track_pose_3d(video_path: str, *, segment: bool, max_frame_count: int | None) -> None:
    mp_pose = mp.solutions.pose  

    rr.log("person", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, timeless=True)
  
    with closing(VideoSource(video_path)) as video_source, mp_pose.Pose() as pose:
        for idx, bgr_frame in enumerate(video_source.stream_bgr()):
            if max_frame_count is not None and idx >= max_frame_count:
                break

            rgb = cv2.cvtColor(bgr_frame.data, cv2.COLOR_BGR2RGB)

            # Associate frame with the data
            rr.set_time_seconds("time", bgr_frame.time)
            rr.set_time_sequence("frame_idx", bgr_frame.idx)

            # Present the video
            rr.log("video/rgb", rr.Image(rgb).compress(jpeg_quality=75))

            # Get the prediction results
            results = pose.process(rgb)
            h, w, _ = rgb.shape

            # New entity "Person" for the 3D presentation
            landmark_positions_3d = read_landmark_positions_3d(results)
            if landmark_positions_3d is not None:
                rr.log(
                    "person/pose/points",
                    rr.Points3D(landmark_positions_3d, class_ids=0, keypoint_ids=mp_pose.PoseLandmark),
                )