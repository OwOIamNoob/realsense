import cv2
import numpy as np
import pyrealsense2 as rs
import json

class Camera:
    def __init__(self, config=None) -> None:
        self.config = rs.config()
        self.pipeline = rs.pipeline()
        self.profile = None
        self.setup()
        
        self.device = self.profile.get_device()
        self.advance = rs.rs400_advanced_mode(self.device)
        if config is not None:
            self.load_config(config)

        self.cam_mtx, self.coef = self.compose_intrinsic_matrix(export_distort=True)
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        self.align = rs.align(rs.stream.color)
        
        self.filters = [rs.disparity_transform(True),
                        rs.temporal_filter(0.4, 80, 3),
                        rs.spatial_filter(0.4, 80, 1, 2),
                        rs.hole_filling_filter(1),
                        rs.disparity_transform(False)]
        self.ema = 0
        self.ema_scale = 0.2
        self.blur_ksize = 3

        pass
    
    def compose_intrinsic_matrix(self, export_distort=False):
        intrinsics = rs.video_stream_profile(self.profile.get_stream(rs.stream.color)).get_intrinsics()
        if not export_distort:
            return np.array([[intrinsics.fx, 0, intrinsics.ppx],
                            [0, intrinsics.fy, intrinsics.ppy],
                            [0, 0, 1]])
        else:
            return np.array([[intrinsics.fx, 0, intrinsics.ppx],
                            [0, intrinsics.fy, intrinsics.ppy],
                            [0, 0, 1]]), intrinsics.coeffs

    def setup(self):
        wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(wrapper)
        device = pipeline_profile.get_device()
        rgb_available = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                rgb_available = True
                break
        if not rgb_available:
            print("Cannot find rgb sensor")
            return Exception("Bruh!!!!")

        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(self.config)
        print(self.profile)
        return True

    def run(self):
        self.ema = 0
        self.pipeline.start()
    
    def stop(self):
        self.pipeline.stop()

    def load_config(self, path):
        file = open(path, "r")
        json_obj = json.load(file)
        if type(next(iter(json_obj))) != str:
            json_obj = {k.encode('utf-8'): v.encode("utf-8") for k, v in json_obj.items()}
        json_string = str(json_obj).replace("'", '\"')
        self.advance.load_json(json_string)

    def numpy_processing(self, frame, ema=False):
        frame = cv2.medianBlur(frame, self.blur_ksize)
        self.ema = self.ema * self.ema_scale + (1 - self.ema_scale) * frame
        
        if ema is True:
            return self.ema.copy()
        else:
            return frame
    
    def post_processing(self, depth_frame):
        
        for filter in self.filters:
            depth_frame = filter.process(depth_frame)

        return depth_frame
    
    def capture(self, postprocess=False):
        frame = self.pipeline.wait_for_frames()
        aligned_frame = self.align.process(frame)
        depth_frame = aligned_frame.get_depth_frame()
        color_frame = aligned_frame.get_color_frame()
        if postprocess is True:
            depth_frame = self.post_processing(depth_frame)
        return depth_frame, color_frame
    
    def to_numpy(self, frame):
        return np.array(frame.get_data())


if __name__ == "__main__":
    config_path = "/home/nguyen/uni/cv/realsense/demo_calib/data/config4.json"
    camera = Camera(config_path)
    while True:
        depth_frame, color_frame = camera.capture(postprocess=True)
        color_img = camera.to_numpy(color_frame)
        depth_img = camera.to_numpy(depth_frame)
        # depth_img = np.interp(depth_img, (depth_img.min(), np.average(depth_img), depth_img.max()), (0, depth_img.max() / 2, depth_img.max())).astype(np.float32)
        depth_img = camera.numpy_processing(depth_img, ema=True)
        # print(depth_img.max())
        depth_cmap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)
        output = np.hstack([color_img, depth_cmap])
        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        cv2.imshow('Align Example', output)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
    
    camera.stop()
        
        