import numpy as np
import os
import pandas as pd


# seq: shape (num_frames, num_hands, num_joints, 3)
def augment_sequence(seq):
    augmented_seq = []
    # pick one rotation + translation per sequence for temporal consistency
    theta = np.deg2rad(np.random.uniform(-30, 30))
    tx, ty = np.random.uniform(-0.05, 0.05, 2)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])
    T = np.array([tx, ty, 0])

    for frame in seq:
        frame_aug = (frame @ R.T) + T
        augmented_seq.append(frame_aug)

    return np.stack(augmented_seq)

def augment_and_save(folder_path):
    for filename in os.listdir(folder_path):
        if not filename.endswith(".csv"):
            continue
        file_path = os.path.join(folder_path, filename)
        print(f"Processing: {filename}")

        df = pd.read_csv(file_path)

        coord_cols = [col for col in df.columns if "_world_" in col]
        coords = df[coord_cols].to_numpy()

        num_frames = coords.shape[0]
        seq = coords.reshape(num_frames, 2, 21, 3)

        aug_seq = augment_sequence(seq)

        # 5️⃣ Flatten back to DataFrame with same column order
        aug_flat = aug_seq.reshape(num_frames, -1)
        aug_df = pd.DataFrame(aug_flat, columns=coord_cols)

        # Optional: copy over metadata columns
        meta_cols = [c for c in df.columns if c not in coord_cols]
        for col in meta_cols:
            aug_df[col] = df[col]

        # 6️⃣ Save to new CSV
        out_path = os.path.join(folder_path, f"aug_{filename}")
        aug_df.to_csv(out_path, index=False)
        print(f"Saved augmented file: {out_path}")

def main(iterations: int, folder_path):
    for i in range(iterations):
        augment_and_save(folder_path)
    