import os
import subprocess

# Where raw videos are stored
raw_videos_folder = "raw_videos"

# Where converted MP4 videos will go
converted_videos_folder = "videos"

# Make sure the output folder exists
os.makedirs(converted_videos_folder, exist_ok=True)

# Loop through each subfolder in raw_videos (e.g., 1, 2, 3)
for subfolder_name in os.listdir(raw_videos_folder):
    subfolder_path = os.path.join(raw_videos_folder, subfolder_name)
    if not os.path.isdir(subfolder_path):
        continue  # skip non-folder files

    # Create matching subfolder in videos/
    out_subfolder_path = os.path.join(converted_videos_folder, subfolder_name)
    os.makedirs(out_subfolder_path, exist_ok=True)

    # Loop through each video in the subfolder
    for video_name in os.listdir(subfolder_path):
        input_path = os.path.join(subfolder_path, video_name)
        if not os.path.isfile(input_path):
            continue  # skip non-file items

        # Output file path with .mp4 extension
        base_name, _ = os.path.splitext(video_name)
        output_path = os.path.join(out_subfolder_path, base_name + ".mp4")

        # Only convert if it doesn't exist yet
        if not os.path.exists(output_path):
            print(f"Converting {input_path} -> {output_path}")
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-y", "-i", input_path,       # input file
                "-r", "30",                   # 30 fps
                "-vf", "scale=1280:720",      # resize to 1280x720
                output_path                   # output file
            ])
        else:
            print(f"Skipping {output_path} (already exists)")
