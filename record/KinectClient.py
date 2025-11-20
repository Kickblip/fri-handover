from pyk4a import (PyK4A, Config, ImageFormat, ColorResolution, DepthMode, FPS,
                   ColorControlCommand, ColorControlMode, PyK4ARecord)

class KinectClient:
    def __init__(self):
        self.config = Config(
            # https://unanancyowen.github.io/k4asdk_python_apireference/classk4a_1_1__bindings_1_1k4atypes_1_1_device_configuration.html
            color_format=ImageFormat.COLOR_MJPG,
            # color_format=ImageFormat.COLOR_BGRA32,
            color_resolution=ColorResolution.RES_1080P,
            depth_mode=DepthMode.NFOV_UNBINNED,
            camera_fps=FPS.FPS_30,
            synchronized_images_only=True,
        )
        self.device = PyK4A(config=self.config, device_id=0)
        self.device.start()

        self.device._set_color_control(cmd=ColorControlCommand.EXPOSURE_TIME_ABSOLUTE,
                      mode=ColorControlMode.MANUAL, value=2500)
        self.device._set_color_control(cmd=ColorControlCommand.WHITEBALANCE,
                            mode=ColorControlMode.MANUAL, value=4500)
        
        self.device._set_color_control(cmd=ColorControlCommand.BRIGHTNESS, mode=ColorControlMode.MANUAL, value=255)
        self.device._set_color_control(cmd=ColorControlCommand.CONTRAST, mode=ColorControlMode.MANUAL, value=10)
        self.device._set_color_control(cmd=ColorControlCommand.SATURATION, mode=ColorControlMode.MANUAL, value=40)
        self.device._set_color_control(cmd=ColorControlCommand.SHARPNESS, mode=ColorControlMode.MANUAL, value=4)
        self.device._set_color_control(cmd=ColorControlCommand.GAIN, mode=ColorControlMode.MANUAL, value=60)

    def start_recording(self, path: str, n_seconds: int = 5) -> None:
        record = PyK4ARecord(device=self.device, config=self.config, path=path)
        record.create()

        print(f"Recording for {n_seconds} seconds...")
        for _ in range(30 * n_seconds):
            capture = self.device.get_capture()
            record.write_capture(capture)

        record.flush()
        record.close()
        print(f"{record.captures_count} frames written.")
            
    def close(self):
        if self.device:
            self.device.stop()
            self.device.close()

