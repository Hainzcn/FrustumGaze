import time
import mediapipe as mp
from utils.image_utils import GlobalImagePreprocessor
from config import settings
from trackers.eye_tracker import EyeTracker
from trackers.face_mesh import FaceMeshTracker, FaceDetectionResultLite
from modules.camera import CameraModel
from .base_process import BaseProcessorProcess


class FrameProcessorProcess(BaseProcessorProcess):

    PROCESS_NAME = "FaceProcessor"

    def __init__(self, input_queue, output_queue, preprocessor, stop_event,
                 shm_names, frame_shape, camera_fov=60.0, triple_buffer_idx=None):
        super().__init__(input_queue, output_queue, stop_event,
                         shm_names, frame_shape, triple_buffer_idx)
        self.preprocessor = preprocessor
        self.camera_fov = camera_fov
        self.last_landmarks_norm = None
        self.using_full_scan = True

    def on_init(self) -> bool:
        try:
            self.face_tracker = FaceMeshTracker()
        except Exception as e:
            print(f"{self.PROCESS_NAME}: 初始化 FaceMeshTracker 失败: {e}")
            return False

        self.tracker = EyeTracker()

        actual_h, actual_w = self.frame_shape[:2]
        camera_model = CameraModel(actual_w, actual_h, self.camera_fov)
        self.cam_matrix = camera_model.cam_matrix
        self.dist_coeffs = camera_model.dist_coeffs
        return True

    def on_process(self, task, frame):
        frame_id = task['frame_id']
        h, w = frame.shape[:2]

        if self.last_landmarks_norm is None:
            if frame_id % settings.FULL_SCAN_INTERVAL != 0:
                return None

        # 预处理
        if self.last_landmarks_norm is None:
            target_h = settings.PREPROCESS_TARGET_HEIGHT
            (target_w, _), _, _ = GlobalImagePreprocessor.calculate_dimensions(frame.shape, target_h)

            resized_bgr = GlobalImagePreprocessor.resize_image(frame, target_size=(target_w, target_h))
            processed_rgb = GlobalImagePreprocessor.to_rgb(resized_bgr)
            processed_rgb = GlobalImagePreprocessor.apply_clahe(processed_rgb)

            scale = target_h / h
            roi_info = (0, 0, w, h, scale)
            self.using_full_scan = True
        else:
            self.using_full_scan = False
            current_padding = 2.0
            if self.preprocessor.last_roi is None:
                current_padding = 3.0

            processed_frame, roi_info = self.preprocessor.process(
                frame, self.last_landmarks_norm, padding_factor=current_padding)
            processed_rgb = GlobalImagePreprocessor.to_rgb(processed_frame)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=processed_rgb)
        timestamp_ms = int(time.time() * 1000)

        detection_result = self.face_tracker.detect(mp_image, timestamp_ms)
        self.preprocessor.restore_landmarks(detection_result, roi_info, w, h)

        if detection_result.face_landmarks:
            if self.last_landmarks_norm is None:
                self.preprocessor.last_roi = None
            self.last_landmarks_norm = detection_result.face_landmarks[0]
        else:
            self.last_landmarks_norm = None

        result_lite = None
        gaze_result = None

        if self.using_full_scan and detection_result.face_landmarks:
            result_lite = FaceDetectionResultLite([])
            self.tracker.reset()
        elif not self.using_full_scan and detection_result.face_landmarks:
            result_lite = FaceDetectionResultLite(detection_result.face_landmarks)

            should_calc_gaze = (frame_id % settings.EYE_GAZE_CALCULATION_INTERVAL == 0)
            face_landmarks = detection_result.face_landmarks[0]

            gaze_result = self.tracker.process_landmarks(
                face_landmarks, w, h, self.camera_fov,
                self.cam_matrix, self.dist_coeffs,
                should_calc_gaze=should_calc_gaze
            )
        else:
            self.tracker.reset()

        return {
            'frame_id': frame_id,
            'detection_result': result_lite,
            'roi_info': roi_info,
            'using_full_scan': self.using_full_scan,
            'timestamp': timestamp_ms,
            'gaze_result': gaze_result
        }

    def on_cleanup(self):
        self.face_tracker.close()
