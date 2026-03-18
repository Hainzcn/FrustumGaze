import time
import mediapipe as mp
from utils.image_utils import GlobalImagePreprocessor
from config import settings
from trackers.pose_tracker import PoseTracker, PoseDetectionResultLite
from .base_process import BaseProcessorProcess


class PoseProcessorProcess(BaseProcessorProcess):

    PROCESS_NAME = "PoseProcessor"

    def on_init(self) -> bool:
        try:
            self.pose_tracker = PoseTracker()
        except Exception as e:
            print(f"{self.PROCESS_NAME}: 初始化 PoseTracker 失败: {e}")
            return False

        (self.target_w, self.target_h), _, _ = \
            GlobalImagePreprocessor.calculate_dimensions(self.frame_shape, settings.POSE_TARGET_HEIGHT)
        return True

    def on_process(self, task, frame):
        frame_id = task['frame_id']

        resized_bgr = GlobalImagePreprocessor.resize_image(frame, target_size=(self.target_w, self.target_h))
        processed_rgb = GlobalImagePreprocessor.to_rgb(resized_bgr)

        timestamp_ms = int(time.time() * 1000)
        pose_landmarks_out = []

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=processed_rgb)
        pose_result = self.pose_tracker.detect(mp_image, timestamp_ms)

        if pose_result.pose_landmarks:
            landmarks = pose_result.pose_landmarks[0]
            for idx in [11, 12, 13, 14]:
                pose_landmarks_out.append(landmarks[idx])

        return {
            'pose_result': PoseDetectionResultLite(pose_landmarks_out),
            'frame_id': frame_id
        }
