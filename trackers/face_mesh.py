
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
from config import settings
from trackers.eye_tracker import EyeTracker
from modules.camera import CameraModel

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
    def __init__(self, input_queue, output_queue, preprocessor, stop_event, shm_names, frame_shape, camera_fov=60.0):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.preprocessor = preprocessor # 注意：preprocessor 会被 pickle 复制到子进程
        self.stop_event = stop_event
        self.shm_names = shm_names # List of names
        self.frame_shape = frame_shape
        self.camera_fov = camera_fov
        self.last_landmarks_norm = None
        self.daemon = True # 设置为守护进程
        self.using_full_scan = True # 初始化状态

    def run(self):
        # --- 在子进程中初始化资源 ---
        
        # 1. 连接共享内存 (双缓冲)
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

        # 2. 初始化 MediaPipe (必须在子进程中进行)
        # 注意：这里假设 model_asset_path 在当前工作目录
        try:
            base_options = python.BaseOptions(model_asset_path=settings.FACE_MESH_TASK_PATH)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1,
                min_face_detection_confidence=settings.FACE_MIN_DETECTION_CONFIDENCE,
                min_face_presence_confidence=settings.FACE_MIN_PRESENCE_CONFIDENCE,
                min_tracking_confidence=settings.FACE_MIN_TRACKING_CONFIDENCE,
                running_mode=vision.RunningMode.VIDEO)
            
            detector = vision.FaceLandmarker.create_from_options(options)
        except Exception as e:
            print(f"FrameProcessorProcess: Failed to init MediaPipe: {e}")
            return

        print("FrameProcessorProcess: Started and Ready.")

        # 3. 初始化 EyeTracker
        tracker = EyeTracker()
        
        # 4. 初始化 CameraModel (用于获取内参)
        # 注意: 这里的 frame_shape 是 (h, w, 3)
        actual_h, actual_w = self.frame_shape[:2]
        camera_model = CameraModel(actual_w, actual_h, self.camera_fov)
        cam_matrix = camera_model.cam_matrix
        dist_coeffs = camera_model.dist_coeffs

        while not self.stop_event.is_set():
            try:
                # 阻塞等待任务
                # 任务格式: {'frame_id': ..., 'buffer_idx': ...}
                task = self.input_queue.get(timeout=0.1) # 增加超时时间防止空轮询过快
                frame_id = task['frame_id']
                buffer_idx = task.get('buffer_idx', 0) # 默认为 0
                
                # 从共享内存直接访问 (Zero-Copy)
                if buffer_idx < len(self.shm_arrays):
                    frame = self.shm_arrays[buffer_idx]
                else:
                    # Fallback
                    frame = self.shm_arrays[0]

                h, w = frame.shape[:2]
                
                # 检查是否需要处理
                should_process = True
                if self.last_landmarks_norm is None:
                     if frame_id % settings.FULL_SCAN_INTERVAL != 0:
                         should_process = False

                if not should_process:
                    continue

                # 预处理：ROI -> 放大 -> 滤波 -> 增强
                # process 内部会进行 crop，因此不会修改原始 shm_array
                
                # 优化逻辑：全图模式下先降分辨率 (BGR) 再转 RGB
                if self.last_landmarks_norm is None:
                    # 全图模式
                    # 1. 降分辨率 (BGR) - 保持与 HandTracker 一致的目标分辨率
                    target_h = settings.PREPROCESS_TARGET_HEIGHT
                    (target_w, _), _, _ = GlobalImagePreprocessor.calculate_dimensions(frame.shape, target_h)
                    
                    resized_bgr = GlobalImagePreprocessor.resize_image(frame, target_size=(target_w, target_h))
                    
                    # 2. 转换 RGB
                    processed_rgb = GlobalImagePreprocessor.to_rgb(resized_bgr)
                    
                    # 3. 增强/滤波
                    processed_rgb = GlobalImagePreprocessor.apply_clahe(processed_rgb) # 保持原有的 CLAHE 增强
                    
                    # 构造 roi_info 以便后续恢复坐标 (x, y, w, h, scale)
                    # 全图缩放模式下:
                    # 原点 (0,0)
                    # 尺寸 (w, h) - 使用原始尺寸，因为 normalized coordinates 相对全图是相同的
                    # 缩放因子 scale = target_h / h
                    scale = target_h / h
                    roi_info = (0, 0, w, h, scale)
                    
                    # 可视化调试：记录当前使用了全图扫描
                    self.using_full_scan = True
                else:
                    self.using_full_scan = False
                    # ROI 模式 (保持原有逻辑，因为 process 内部已经做了裁剪和缩放)
                    # 动态调整 padding: 如果是刚从全图扫描恢复（last_roi 为 None），使用更大的 padding 以确保捕捉到目标
                    current_padding = 2.0
                    if self.preprocessor.last_roi is None:
                        current_padding = 3.0
                    
                    processed_frame, roi_info = self.preprocessor.process(frame, self.last_landmarks_norm, padding_factor=current_padding)
                    
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
                    # 如果之前是全图模式（last_landmarks_norm is None），说明刚刚找回目标
                    if self.last_landmarks_norm is None:
                        # 强制重置 ROI 平滑器的状态，避免与旧 ROI 平滑导致裁剪不准
                        self.preprocessor.last_roi = None
                        
                    self.last_landmarks_norm = detection_result.face_landmarks[0]
                else:
                    self.last_landmarks_norm = None
                
                # 优化: 全图扫描模式下，检测到目标后仅返回 ROI 区域信息，不返回详细的关键点数据
                # 这样可以避免 Main 进程进行昂贵的 EyeTracking 计算
                result_lite = None
                processed_gaze_data = None
                
                if self.using_full_scan and detection_result.face_landmarks:
                     # 构造一个空的 Result，或者包含特定标志
                     result_lite = DetectionResultLite([]) # 空的关键点列表
                     tracker.reset() # 丢失跟踪时重置滤波器
                elif not self.using_full_scan and detection_result.face_landmarks:
                    # 正常 ROI 模式，返回完整结果
                    result_lite = DetectionResultLite(detection_result.face_landmarks)
                    
                    # --- 执行视线解算 ---
                    should_calc_gaze = (frame_id % settings.EYE_GAZE_CALCULATION_INTERVAL == 0)
                    
                    # 只需要处理第一个人脸
                    face_landmarks = detection_result.face_landmarks[0]
                    
                    processed_gaze_data = tracker.process_landmarks(
                        face_landmarks, w, h, self.camera_fov, cam_matrix, dist_coeffs,
                        should_calc_gaze=should_calc_gaze
                    )
                    
                    # 添加额外的 tracker 状态信息以便主进程直接使用
                    if processed_gaze_data:
                        est_dist, off_x, off_y = tracker.get_gaze_params()
                        processed_gaze_data['gaze_params'] = (est_dist, off_x, off_y)
                        processed_gaze_data['head_center_pos'] = tracker.head_center_pos
                        processed_gaze_data['current_pixel_dist'] = tracker.current_pixel_dist
                else:
                    tracker.reset()

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
                    'using_full_scan': self.using_full_scan, # 传递全图扫描状态
                    'timestamp': timestamp_ms,
                    'processed_gaze_data': processed_gaze_data # 新增：处理后的视线数据
                })
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing Error in Process: {e}")
                import traceback
                traceback.print_exc()

        # 清理
        for mgr in self.shm_managers:
            try:
                mgr.close()
            except:
                pass
