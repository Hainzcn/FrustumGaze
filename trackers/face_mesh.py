
import multiprocessing
import queue
import time
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from modules.shared_mem import get_shared_array
from utils.image_utils import GlobalImagePreprocessor

# 定义简单的 Landmark 类以便于 Pickle
class LandmarkLite:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

# 定义简单的 Result 类以便于 Pickle，避免直接传递 MediaPipe 复杂对象
class DetectionResultLite:
    def __init__(self, face_landmarks_list):
        self.face_landmarks = []
        # 将 NormalizedLandmark 对象转换为简单的 (x, y, z) 字典或对象
        if face_landmarks_list:
            for landmarks in face_landmarks_list:
                simple_landmarks = []
                for lm in landmarks:
                    simple_landmarks.append(LandmarkLite(lm.x, lm.y, lm.z))
                self.face_landmarks.append(simple_landmarks)

class FrameProcessorProcess(multiprocessing.Process):
    def __init__(self, input_queue, output_queue, preprocessor, stop_event, shm_name, frame_shape):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.preprocessor = preprocessor # 注意：preprocessor 会被 pickle 复制到子进程
        self.stop_event = stop_event
        self.shm_name = shm_name
        self.frame_shape = frame_shape
        self.last_landmarks_norm = None
        self.daemon = True # 设置为守护进程

    def run(self):
        # --- 在子进程中初始化资源 ---
        
        # 1. 连接共享内存
        try:
            shm_manager, shm_array = get_shared_array(self.shm_name, self.frame_shape)
        except Exception as e:
            print(f"FrameProcessorProcess: Failed to connect to shared memory: {e}")
            return

        # 2. 初始化 MediaPipe (必须在子进程中进行)
        # 注意：这里假设 model_asset_path 在当前工作目录
        try:
            base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                running_mode=vision.RunningMode.VIDEO)
            
            detector = vision.FaceLandmarker.create_from_options(options)
        except Exception as e:
            print(f"FrameProcessorProcess: Failed to init MediaPipe: {e}")
            return

        print("FrameProcessorProcess: Started and Ready.")

        while not self.stop_event.is_set():
            try:
                # 阻塞等待任务
                # 任务格式: {'frame_id': ...}
                task = self.input_queue.get(timeout=0.1) # 增加超时时间防止空轮询过快
                frame_id = task['frame_id']
                
                # 从共享内存直接访问 (Zero-Copy)
                frame = shm_array
                h, w = frame.shape[:2]
                
                # 预处理：ROI -> 放大 -> 滤波 -> 增强
                # process 内部会进行 crop，因此不会修改原始 shm_array
                processed_frame, roi_info = self.preprocessor.process(frame, self.last_landmarks_norm)
                
                # 转换处理后的帧为 RGB (MediaPipe 需要)
                # 使用全局工具
                processed_rgb = GlobalImagePreprocessor.to_rgb(processed_frame)
                
                # MediaPipe 处理
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=processed_rgb)
                timestamp_ms = int(time.time() * 1000)
                
                detection_result = detector.detect_for_video(mp_image, timestamp_ms)
                
                # 坐标还原 (Local Normalized -> Global Normalized)
                # ... (保持原有逻辑)

                self.preprocessor.restore_landmarks(detection_result, roi_info, w, h)
                
                # 更新上一帧 Landmarks (用于下一帧 ROI 计算)
                if detection_result.face_landmarks:
                    self.last_landmarks_norm = detection_result.face_landmarks[0]
                else:
                    self.last_landmarks_norm = None
                
                # 将结果转换为轻量级对象以便传输
                result_lite = DetectionResultLite(detection_result.face_landmarks)
                
                # 将结果放入输出队列
                if self.output_queue.full():
                    try:
                        self.output_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.output_queue.put({
                    'frame_id': frame_id,
                    'detection_result': result_lite,
                    'roi_info': roi_info,
                    'timestamp': timestamp_ms
                })
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing Error in Process: {e}")
                import traceback
                traceback.print_exc()

        # 清理
        shm_manager.close()
