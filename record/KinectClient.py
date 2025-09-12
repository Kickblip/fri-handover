from pyk4a import Config, ImageFormat, PyK4A, PyK4ARecord

class KinectClient:
    def __init__(self):
        self.config = Config(color_format=ImageFormat.COLOR_MJPG)
        self.device = PyK4A(config=self.config, device_id=0)
        self.device.start()
        
    def start_recording(self, path: str, n_seconds: int = 5) -> None:
        print(f"Open record file {path}")
        record = PyK4ARecord(device=self.device, config=self.config, path=path)
        record.create()

        print(f"Recording for {n_seconds} seconds... Press CTRL-C to stop recording.")
        for _ in range(30 * n_seconds):
            capture = self.device.get_capture()
            record.write_capture(capture)

        record.flush()
        record.close()
        print(f"{record.captures_count} frames written.")



# from argparse import ArgumentParser

# from pyk4a import Config, ImageFormat, PyK4A, PyK4ARecord

# parser = ArgumentParser(description="pyk4a recorder")
# parser.add_argument("--device", type=int, help="Device ID", default=0)
# parser.add_argument("FILE", type=str, help="Path to MKV file")
# args = parser.parse_args()

# print(f"Starting device #{args.device}")
# config = Config(color_format=ImageFormat.COLOR_MJPG)
# device = PyK4A(config=config, device_id=0)
# device.start()

# print(f"Open record file {args.FILE}")
# record = PyK4ARecord(device=device, config=config, path=args.FILE)
# record.create()
# try:
#     print("Recording... Press CTRL-C to stop recording.")
#     while True:
#         capture = device.get_capture()
#         record.write_capture(capture)
# except KeyboardInterrupt:
#     print("CTRL-C pressed. Exiting.")

# record.flush()
# record.close()
# print(f"{record.captures_count} frames written.")