import time
import math
import numpy as np
import mediapipe as mp
from utils.image_utils import GlobalImagePreprocessor
from config import settings
from trackers.hand_tracker import HandTracker, HandDetectionResultLite
from trackers.common import LandmarkLite
from .base_process import BaseProcessorProcess


class HandProcessorProcess(BaseProcessorProcess):

    PROCESS_NAME = "HandProcessor"

    def __init__(self, input_queue, output_queue, stop_event,
                 shm_names, frame_shape, fov=60.0, triple_buffer_idx=None):
        super().__init__(input_queue, output_queue, stop_event,
                         shm_names, frame_shape, triple_buffer_idx)
        self.fov = fov

    def on_init(self) -> bool:
        try:
            self.tracker = HandTracker(fov=self.fov)
        except Exception as e:
            print(f"{self.PROCESS_NAME}: 初始化 HandTracker 失败: {e}")
            return False

        (self.target_w, self.target_h), self.global_scale, _ = \
            GlobalImagePreprocessor.calculate_dimensions(self.frame_shape, settings.PREPROCESS_TARGET_HEIGHT)
        self.aspect_ratio = self.frame_shape[1] / float(self.frame_shape[0])

        focal_length = (self.target_w / 2.0) / math.tan(math.radians(self.fov) / 2.0)
        center = (self.target_w / 2.0, self.target_h / 2.0)
        self.camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        return True

    def on_process(self, task, frame):
        frame_id = task['frame_id']
        h, w = frame.shape[:2]
        target_w, target_h = self.target_w, self.target_h

        timestamp_ms = int(time.time() * 1000)

        # --- ROI 处理逻辑 ---
        roi_info = None
        processed_rgb = None
        should_process_hand = True

        if self.tracker.roi:
            cropped_roi, roi_rect = GlobalImagePreprocessor.crop_by_normalized_roi(frame, self.tracker.roi)
            if cropped_roi is not None:
                resized_roi = GlobalImagePreprocessor.resize_image(
                    cropped_roi, scale_factor=settings.PREPROCESS_ROI_SCALE_FACTOR)
                processed_rgb = GlobalImagePreprocessor.to_rgb(resized_roi)
                roi_info = roi_rect
            else:
                self.tracker.roi = None
                self.tracker.roi_miss_count = 0

        need_full_frame = (not self.tracker.roi and frame_id % settings.FULL_SCAN_INTERVAL == 0)
        processed_rgb_full = None

        if need_full_frame:
            resized_bgr = GlobalImagePreprocessor.resize_image(frame, target_size=(target_w, target_h))
            processed_rgb_full = GlobalImagePreprocessor.to_rgb(resized_bgr)
            processed_rgb_full = GlobalImagePreprocessor.apply_gaussian_blur(
                processed_rgb_full,
                kernel_size=settings.PREPROCESS_GAUSSIAN_KERNEL_SIZE,
                sigma=settings.PREPROCESS_GAUSSIAN_SIGMA)

        if not self.tracker.roi:
            if processed_rgb_full is not None:
                processed_rgb = processed_rgb_full
            else:
                should_process_hand = False

        mapped_landmarks_list = []
        detection_result = None

        if should_process_hand and processed_rgb is not None:
            if processed_rgb is not processed_rgb_full:
                processed_rgb = GlobalImagePreprocessor.apply_gaussian_blur(
                    processed_rgb,
                    kernel_size=settings.PREPROCESS_GAUSSIAN_KERNEL_SIZE,
                    sigma=settings.PREPROCESS_GAUSSIAN_SIGMA)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=processed_rgb)
            detection_result = self.tracker.process(mp_image, timestamp_ms)

            if detection_result.hand_landmarks:
                self.tracker.roi_miss_count = 0

                for landmarks in detection_result.hand_landmarks:
                    mapped_landmarks = []
                    for lm in landmarks:
                        h_curr, w_curr = processed_rgb.shape[:2]
                        px = lm.x * w_curr
                        py = lm.y * h_curr

                        if roi_info:
                            px = px / settings.PREPROCESS_ROI_SCALE_FACTOR
                            py = py / settings.PREPROCESS_ROI_SCALE_FACTOR
                            roi_x, roi_y, _, _ = roi_info
                            px += roi_x
                            py += roi_y
                            final_x = (px / w) * target_w
                            final_y = (py / h) * target_h
                        else:
                            final_x = px
                            final_y = py

                        norm_x = final_x / target_w
                        norm_y = final_y / target_h
                        mapped_landmarks.append(LandmarkLite(norm_x, norm_y, lm.z))
                    mapped_landmarks_list.append(mapped_landmarks)

                next_roi = self.tracker.calculate_roi(mapped_landmarks_list)
                if next_roi:
                    self.tracker.roi = next_roi
            else:
                self.tracker.roi_miss_count += 1
                mapped_landmarks_list = []
                if self.tracker.roi_miss_count > self.tracker.MAX_ROI_MISS_COUNT:
                    self.tracker.roi = None

        # 过滤重叠手部
        filtered_landmarks = []
        filtered_handedness = []

        if should_process_hand and detection_result is not None:
            filtered_landmarks, filtered_handedness = self.tracker.filter_overlapping_hands(
                mapped_landmarks_list,
                detection_result.handedness
            )

        result_lite = HandDetectionResultLite(filtered_landmarks, filtered_handedness)

        closest_hand_info = None
        min_z = float('inf')
        hands_pos = []

        if result_lite.multi_hand_landmarks:
            for idx, landmarks in enumerate(result_lite.multi_hand_landmarks):
                label = "Unknown"
                if result_lite.multi_handedness and idx < len(result_lite.multi_handedness):
                    categories = result_lite.multi_handedness[idx]
                    if categories:
                        label = categories[0]['label']

                x, y, z, w_norm, yaw, pitch, motion_score, grip_factor, depth_details, pinch_result = \
                    self.tracker.calculate_hand_pos(
                        landmarks, target_w, target_h, self.aspect_ratio,
                        timestamp=timestamp_ms / 1000.0,
                        camera_matrix=self.camera_matrix,
                        hand_label=label,
                        frame_id=frame_id
                    )

                if x is not None:
                    is_pinching, px, py, pz, pinch_cx, pinch_cy = pinch_result

                    hand_info = {
                        'id': idx, 'label': label,
                        'x': x, 'y': y, 'z': z,
                        'yaw': yaw, 'pitch': pitch,
                        'w_norm': w_norm,
                        'motion_score': motion_score,
                        'grip_factor': grip_factor,
                        'depth_details': depth_details,
                        'landmarks': landmarks,
                        'is_pinching': is_pinching,
                        'pinch_pos': (px, py, pz),
                        'pinch_center_2d': (pinch_cx, pinch_cy)
                    }
                    hands_pos.append(hand_info)

                    if z < min_z:
                        min_z = z
                        closest_hand_info = hand_info

        return {
            'frame_id': frame_id,
            'hand_result': result_lite,
            'timestamp': timestamp_ms,
            'closest_hand': closest_hand_info,
            'hands_pos': hands_pos
        }

    def on_cleanup(self):
        self.tracker.close()
