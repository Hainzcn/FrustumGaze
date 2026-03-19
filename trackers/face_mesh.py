import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from config import settings
from .common import LandmarkLite

class FaceDetectionResultLite:
    def __init__(self, face_landmarks_list, face_blendshapes_list=None):
        self.face_landmarks = []
        self.eye_blink_left = 0.0
        self.eye_blink_right = 0.0

        if face_landmarks_list:
            for landmarks in face_landmarks_list:
                simple_landmarks = []
                for lm in landmarks:
                    simple_landmarks.append(LandmarkLite(lm.x, lm.y, lm.z))
                self.face_landmarks.append(simple_landmarks)

        if face_blendshapes_list and len(face_blendshapes_list) > 0:
            for bs in face_blendshapes_list[0]:
                name = bs.category_name
                if name == "eyeBlinkLeft":
                    self.eye_blink_left = bs.score
                elif name == "eyeBlinkRight":
                    self.eye_blink_right = bs.score

class FaceMeshTracker:
    def __init__(self):
        self.detector = None
        self._init_mediapipe()

    def _init_mediapipe(self):
        try:
            base_options = python.BaseOptions(model_asset_path=settings.FACE_MESH_TASK_PATH)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=False,
                num_faces=1,
                min_face_detection_confidence=settings.FACE_MIN_DETECTION_CONFIDENCE,
                min_face_presence_confidence=settings.FACE_MIN_PRESENCE_CONFIDENCE,
                min_tracking_confidence=settings.FACE_MIN_TRACKING_CONFIDENCE,
                running_mode=vision.RunningMode.VIDEO)
            
            self.detector = vision.FaceLandmarker.create_from_options(options)
            print("FaceMeshTracker: MediaPipe 初始化完成。")
        except Exception as e:
            print(f"FaceMeshTracker: MediaPipe 初始化失败: {e}")
            raise e

    def detect(self, mp_image, timestamp_ms):
        if not self.detector:
            return None
        return self.detector.detect_for_video(mp_image, timestamp_ms)

    def close(self):
        if self.detector:
            self.detector.close()
