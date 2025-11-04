import cv2
import pandas as pd
import numpy as np

# Camera intrinsics — adjust if your setup differs
fx, fy = 600, 600
cx, cy = 960, 540  # center for 1920x1080 video

def project_to_image(X, Y, Z):
    """Project 3D world points to 2D pixel coordinates."""
    if Z <= 0:
        return None
    x_pix = (fx * X / Z) + cx
    y_pix = (fy * Y / Z) + cy
    return int(x_pix), int(y_pix)

def overlay_csv_on_video(csv_path, video_path, out_path=None, show=True):
    df = pd.read_csv(csv_path)
    coord_cols = [c for c in df.columns if "_world_" in c]
    coords = df[coord_cols].to_numpy()

    num_frames = coords.shape[0]
    try:
        coords = coords.reshape(num_frames, 2, 21, 3)
    except ValueError:
        print(f"⚠️ {csv_path}: coordinate count mismatch ({len(coord_cols)} columns).")
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = None
    if out_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    colors = [(0, 0, 255), (255, 0, 0)]  # red, blue
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx >= num_frames:
            break

        for hand_i, hand in enumerate(coords[frame_idx]):
            for (X, Y, Z) in hand:
                if np.isnan(X) or np.isnan(Y) or np.isnan(Z) or Z <= 0:
                    continue
                uv = project_to_image(X, Y, Z)
                if uv is None:
                    continue
                u, v = uv
                if 0 <= u < width and 0 <= v < height:
                    cv2.circle(frame, (u, v), 4, colors[hand_i], -1, lineType=cv2.LINE_AA)

        if show:
            cv2.imshow("Overlay", frame)
            if cv2.waitKey(int(1000/fps)) & 0xFF == 27:
                break

        if out:
            out.write(frame)

        frame_idx += 1

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print(f"✅ Overlay complete: {out_path if out_path else '(not saved)'}")

def main():
    overlay_csv_on_video(
        "/Users/diego/Programming/fri-handover/dataset/mediapipe_outputs/csv/world/aug_1_w_b_world.csv",
        "/Users/diego/Programming/fri-handover/dataset/mediapipe_outputs/video/world/2_w_b_world.mp4",
        out_path="/Users/diego/Programming/fri-handover/overlay1.mp4",
        show=True
    )

if __name__ == "__main__":
    main()
