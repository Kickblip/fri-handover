```
# Run all steps for all videos
python mediapipe_runner.py

# Run only Step 1 (MediaPipe world extraction)
python mediapipe_runner.py --step 1

# Run Step 2 ( world → quaternion)
python mediapipe_runner.py --step 2

# Run Step 3 only for one video (quaternion → Rodrigues)
python mediapipe_runner.py --step 3 --video 3_handover.mkv

```