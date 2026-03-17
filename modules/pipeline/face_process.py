import multiprocessing
import queue
import time
import signal
import mediapipe as mp
import numpy as np
from modules.shared_mem import get_shared_array
from utils.image_utils import GlobalImagePreprocessor
from config import settings
from trackers.eye_tracker import EyeTracker
from trackers.face_mesh import FaceMeshTracker, FaceDetectionResultLite
from modules.camera import CameraModel

class FrameProcessorProcess(multiprocessing.Process):
    def __init__(self, input_queue, output_queue, preprocessor, stop_event, shm_names, frame_shape, camera_fov=60.0, triple_buffer_idx=None):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.preprocessor = preprocessor # 注意：preprocessor 会被 pickle 复制到子进程
        self.stop_event = stop_event
        self.shm_names = shm_names # List of names
        self.frame_shape = frame_shape
        self.camera_fov = camera_fov
        self.triple_buffer_idx = triple_buffer_idx # 三缓冲原子索引
        self.last_landmarks_norm = None
        self.daemon = True # 设置为守护进程
        self.using_full_scan = True # 初始化状态

    def run(self):
        # 忽略 SIGINT 信号，让主进程处理 Ctrl+C
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        # --- 在子进程中初始化资源 ---
        
        # 1. 连接共享内存 (三缓冲)
        self.shm_managers = []
        self.shm_arrays = []
        
        # 兼容旧代码传入单个 name 的情况 (虽然不建议)
        names = self.shm_names if isinstance(self.shm_names, list) else [self.shm_names]
        
        for name in names:
            try:
                mgr, arr = get_shared_array(name, self.frame_shape)
                self.shm_managers.append(mgr)
                self.shm_arrays.append(arr)
            except Exception as e:
                print(f"FrameProcessorProcess: Failed to connect to shared memory {name}: {e}")
                return

        # 2. 初始化 FaceMeshTracker
        try:
            face_tracker = FaceMeshTracker()
        except Exception as e:
            print(f"FrameProcessorProcess: 初始化 FaceMeshTracker 失败: {e}")
            return

        print("FrameProcessorProcess: 进程已启动并就绪。")

        # 3. 初始化 EyeTracker
        tracker = EyeTracker()
        
        # 4. 初始化 CameraModel (用于获取内参)
        actual_h, actual_w = self.frame_shape[:2]
        camera_model = CameraModel(actual_w, actual_h, self.camera_fov)
        cam_matrix = camera_model.cam_matrix
        dist_coeffs = camera_model.dist_coeffs

        while not self.stop_event.is_set():
            try:
                # 非阻塞检查停止事件，带超时
                if self.stop_event.is_set():
                    break
                
                try:
                    # 阻塞等待任务，设置短超时以便定期检查 stop_event
                    task = self.input_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                frame_id = task['frame_id']
                
                # 三缓冲：始终从最近写完的 buffer 读取
                if self.triple_buffer_idx is not None:
                    read_idx = self.triple_buffer_idx.value
                else:
                    read_idx = task.get('buffer_idx', 0)
                
                if 0 <= read_idx < len(self.shm_arrays):
                    frame = self.shm_arrays[read_idx]
                else:
                    frame = self.shm_arrays[0]

                h, w = frame.shape[:2]
                
                # 检查是否需要处理
                should_process = True
                if self.last_landmarks_norm is None:
                    if frame_id % settings.FULL_SCAN_INTERVAL != 0:
                        should_process = False

                if not should_process:
                    continue

                # 预处理
                if self.last_landmarks_norm is None:
                    # 全图模式
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
                    
                    processed_frame, roi_info = self.preprocessor.process(frame, self.last_landmarks_norm, padding_factor=current_padding)
                    processed_rgb = GlobalImagePreprocessor.to_rgb(processed_frame)
                
                # MediaPipe 处理
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=processed_rgb)
                timestamp_ms = int(time.time() * 1000)
                
                detection_result = face_tracker.detect(mp_image, timestamp_ms)
                
                # 坐标还原
                self.preprocessor.restore_landmarks(detection_result, roi_info, w, h)
                
                # 更新上一帧 Landmarks
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
                    tracker.reset()
                elif not self.using_full_scan and detection_result.face_landmarks:
                    result_lite = FaceDetectionResultLite(detection_result.face_landmarks)
                    
                    should_calc_gaze = (frame_id % settings.EYE_GAZE_CALCULATION_INTERVAL == 0)
                    face_landmarks = detection_result.face_landmarks[0]
                    
                    gaze_result = tracker.process_landmarks(
                        face_landmarks, w, h, self.camera_fov, cam_matrix, dist_coeffs,
                        should_calc_gaze=should_calc_gaze
                    )
                else:
                    tracker.reset()

                if self.output_queue.full():
                    try:
                        self.output_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.output_queue.put({
                    'frame_id': frame_id,
                    'detection_result': result_lite,
                    'roi_info': roi_info,
                    'using_full_scan': self.using_full_scan,
                    'timestamp': timestamp_ms,
                    'gaze_result': gaze_result
                })
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing Error in Process: {e}")
                import traceback
                traceback.print_exc()

        # 清理
        face_tracker.close()
        for mgr in self.shm_managers:
            try:
                mgr.close()
            except:
                pass
