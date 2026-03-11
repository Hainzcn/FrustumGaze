import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from config import settings
from .common import LandmarkLite

class PoseDetectionResultLite:
    def __init__(self, pose_landmarks=None):
        self.pose_landmarks = [] 
        if pose_landmarks:
            for lm in pose_landmarks:
                 vis = getattr(lm, 'visibility', 1.0)
                 self.pose_landmarks.append(LandmarkLite(lm.x, lm.y, lm.z, vis))

class PoseTracker:
    def __init__(self):
        self.detector = None
        self._init_mediapipe()

    def _init_mediapipe(self):
        try:
            base_options_pose = python.BaseOptions(model_asset_path=settings.POSE_LANDMARKER_TASK_PATH)
            options_pose = vision.PoseLandmarkerOptions(
                base_options=base_options_pose,
                num_poses=1,
                min_pose_detection_confidence=settings.POSE_MIN_DETECTION_CONFIDENCE,
                min_pose_presence_confidence=settings.POSE_MIN_PRESENCE_CONFIDENCE,
                min_tracking_confidence=settings.POSE_MIN_TRACKING_CONFIDENCE,
                running_mode=vision.RunningMode.VIDEO)
            self.detector = vision.PoseLandmarker.create_from_options(options_pose)
            print(f"PoseTracker: MediaPipe Initialized.")
        except Exception as e:
            print(f"PoseTracker: Failed to init MediaPipe: {e}")
            raise e

    def detect(self, mp_image, timestamp_ms):
        if not self.detector:
            return None
        return self.detector.detect_for_video(mp_image, timestamp_ms)

    def close(self):
        if self.detector:
            self.detector.close()
