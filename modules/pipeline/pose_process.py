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
        return True

    def on_process(self, task, frame):
        frame_id = task['frame_id']
        h, w = frame.shape[:2]

        (target_w, target_h), _, _ = GlobalImagePreprocessor.calculate_dimensions(
            frame.shape, settings.PREPROCESS_TARGET_HEIGHT)

        resized_bgr = GlobalImagePreprocessor.resize_image(frame, target_size=(target_w, target_h))
        processed_rgb = GlobalImagePreprocessor.to_rgb(resized_bgr)
        processed_rgb = GlobalImagePreprocessor.apply_gaussian_blur(
            processed_rgb,
            kernel_size=settings.PREPROCESS_GAUSSIAN_KERNEL_SIZE,
            sigma=settings.PREPROCESS_GAUSSIAN_SIGMA)

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
