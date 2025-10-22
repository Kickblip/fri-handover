```
# Run all steps for all videos
python run_pipeline.py

# Run only Step 1 (MediaPipe world extraction)
python run_pipeline.py --step 1

# Run Step 3 (quaternion → Rodrigues)
python run_pipeline.py --step 3

# Run Step 4 only for one video
python run_pipeline.py --step 4 --video 3_handover.mkv

```