import cv2
import numpy as np
from pyk4a import PyK4APlayback
from argparse import ArgumentParser
from helpers import colorize

def main():
    parser = ArgumentParser(description="Depth/Color viewer")
    parser.add_argument("FILE", type=str, help="Path to MKV file")
    args = parser.parse_args()

    playback = PyK4APlayback(args.FILE)
    playback.open()

    cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)
    
    try:
        while True:
            try:
                capture = playback.get_next_capture()
            except EOFError:
                break

            if np.any(capture.color):
                cv2.imshow("RGB", capture.color[:, :, :3])

            if np.any(capture.depth):
                cv2.imshow("Depth", colorize(capture.depth, (None, 5000), cv2.COLORMAP_HSV))
                
            key = cv2.waitKey(10)
            if key != -1:
                cv2.destroyAllWindows()
                break
    finally:
        cv2.destroyAllWindows()
        playback.close()

if __name__ == "__main__":
    main()