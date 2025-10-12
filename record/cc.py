import os
from pathlib import Path
import argparse
import cv2
import numpy as np

parser = argparse.ArgumentParser(prog='Color Correction',)

parser.add_argument('filename')
args = parser.parse_args()
path = args.filename

if os.path.isfile(path):
    print("Target File:", path)
base_dir = Path(__file__).parent.parent  
full_path = os.path.join(base_dir, path)


cap = cv2.VideoCapture(full_path, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise RuntimeError("Failed to open file")

def prepare_for_mediapipe(frame_bgr):
    yuv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV)
    y,u,v = cv2.split(yuv)
    y = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(y)
    yuv = cv2.merge([y,u,v])
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    gamma = 0.9
    inv = 1.0/gamma
    lut = (np.arange(256)/255.0)**inv * 255
    bgr = cv2.LUT(bgr, lut.astype(np.uint8))

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

try:
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
 
        frame_bgr = prepare_for_mediapipe(frame_bgr)

        cv2.imshow("Color Corrected", frame_bgr)
        cv2.waitKey(1)

finally:
    cap.release()
    cv2.destroyAllWindows()
