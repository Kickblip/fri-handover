```
# Run all steps for all videos
python mediapipe_runner.py

# Run only Step 1 (MediaPipe world extraction)
python mediapipe_runner.py --step 1

# Run Step 3 (quaternion → Rodrigues)
python mediapipe_runner.py --step 3

# Run Step 4 only for one video
python mediapipe_runner.py --step 4 --video 3_handover.mkv

```