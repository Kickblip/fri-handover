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

venv) (base) bwilab@hamilton:~/Documents/fri-handover$ python -m model.train
🟢 Device: cuda
Traceback (most recent call last):
  File "/usr/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/usr/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/home/bwilab/Documents/fri-handover/model/train.py", line 117, in <module>
    train()
  File "/home/bwilab/Documents/fri-handover/model/train.py", line 43, in train
    train_ld, val_ld, test_ld = build_loaders()
  File "/home/bwilab/Documents/fri-handover/model/data.py", line 151, in build_loaders
    return mk(tr, True), mk(va, False), mk(te, False)
  File "/home/bwilab/Documents/fri-handover/model/data.py", line 149, in <lambda>
    mk = lambda ss, shuf: DataLoader(HandoverDataset(ss, SEQ_LEN, SEQ_STRIDE),
  File "/home/bwilab/Documents/fri-handover/model/data.py", line 130, in __init__
    X, frames = load_features(s)
  File "/home/bwilab/Documents/fri-handover/model/data.py", line 85, in load_features
    X_h, frames = load_rodrigues(stem)
  File "/home/bwilab/Documents/fri-handover/model/data.py", line 46, in load_rodrigues
    hcol = _pick_col(df, "hand",  ["which", "side"])
  File "/home/bwilab/Documents/fri-handover/model/data.py", line 29, in _pick_col
    raise KeyError(f"Need one of {[main]+aliases} in columns: {list(df.columns)[:8]} ...")
KeyError: "Need one of ['hand', 'which', 'side'] in columns: ['time_sec', 'frame_index', 'hand_label_0', 'hand_label_1', 'hand_score_0', 'hand_score_1', 'rot_vec_x_0', 'rot_vec_y_0'] ..."
(venv) (base) bwilab@hamilton:~/Documents/fri-handover$ 