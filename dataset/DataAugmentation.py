import numpy as np
import os
import pandas as pd
import argparse

# seq: shape (num_frames, num_hands, num_joints, 3)
def augment_sequence(seq):
    augmented_seq = []
    # pick one rotation + translation per sequence for temporal consistency
    theta = np.deg2rad(np.random.uniform(-15, 15))
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

        df = pd.read_csv(file_path, dtype={'hand_label_0': str, 'hand_label_1': str})
        df = df.reset_index(drop=True)  # Ensure sequential row indices

        coord_cols = [col for col in df.columns if "_vec_" in col]
        coords = df[coord_cols].to_numpy()

        num_frames = coords.shape[0]
        seq = coords.reshape(num_frames, -1, 3)
        aug_seq = augment_sequence(seq)
        aug_flat = aug_seq.reshape(num_frames, -1)

        # overwrite in place (preserves order)
        df.loc[:, coord_cols] = aug_flat

        out_path = os.path.join(folder_path, f"aug_{filename}")
        df.to_csv(out_path, index=False, float_format="%.17f")

        print(f"Saved augmented file: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Augment hand 3D coordinate CSVs. Make sure columns in csv are in _vec_ format.")
    parser.add_argument(
        "--csv_dir",
        type=str,
        required=True,
        help="Directory containing input Mediapipe CSV files."
    )

    args = parser.parse_args()

    print("=== Starting augmentation ===")
    augment_and_save(args.csv_dir)
    print("=== Augmentation complete ===")


if __name__ == "__main__":
    main()
    