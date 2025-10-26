Train the model
```
python -m handover.train
```
Run inference 
```
python -m handover.infer 1_video --out dataset/mediapipe_outputs/csv/rodrigues/1_video_probs.csv
```

Visualize with OpenGL (Open3D)
```
python viz_open3d.py 1_video
```