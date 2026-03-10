import multiprocessing
import queue
import time
import math
import numpy as np
import mediapipe as mp
from modules.shared_mem import get_shared_array
from utils.image_utils import GlobalImagePreprocessor
from config import settings
from trackers.hand_tracker import HandTracker, HandDetectionResultLite
from trackers.common import LandmarkLite

class HandProcessorProcess(multiprocessing.Process):
    def __init__(self, input_queue, output_queue, stop_event, shm_names, frame_shape, fov=60.0):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.shm_names = shm_names # List of names
        self.frame_shape = frame_shape
        self.fov = fov
        self.daemon = True

    def run(self):
        # --- 在子进程中初始化资源 ---
        
        # 1. 连接共享内存 (双缓冲)
        self.shm_managers = []
        self.shm_arrays = []
        
        # 兼容旧代码传入单个 name 的情况
        names = self.shm_names if isinstance(self.shm_names, list) else [self.shm_names]
        
        for name in names:
            try:
                mgr, arr = get_shared_array(name, self.frame_shape)
                self.shm_managers.append(mgr)
                self.shm_arrays.append(arr)
            except Exception as e:
                print(f"HandProcessorProcess: Failed to connect to shared memory {name}: {e}")
                return

        # 2. 初始化 HandTracker
        try:
            tracker = HandTracker(fov=self.fov)
        except Exception as e:
            print(f"HandProcessorProcess: Failed to init HandTracker: {e}")
            return
        
        print(f"HandProcessorProcess: Started and Ready. FOV={self.fov}")

        # 初始化缓存
        cached_dims = (0, 0)
        cached_camera_matrix = None

        while not self.stop_event.is_set():
            try:
                # 阻塞等待任务
                task = self.input_queue.get(timeout=0.1)
                frame_id = task['frame_id']
                buffer_idx = task.get('buffer_idx', 0)
                
                # 从共享内存复制图像数据
                if buffer_idx < len(self.shm_arrays):
                    frame = self.shm_arrays[buffer_idx]
                else:
                    frame = self.shm_arrays[0]
                
                # 获取原始分辨率
                h, w = frame.shape[:2]
                aspect_ratio = w / float(h)
                
                # 计算全图模式下的目标分辨率 (用于 PnP 和全图扫描)
                (target_w, target_h), global_scale, _ = GlobalImagePreprocessor.calculate_dimensions(frame.shape, settings.PREPROCESS_TARGET_HEIGHT)
                
                # 优化：预计算/缓存相机矩阵 (基于 720p 目标分辨率)
                if (target_w, target_h) != cached_dims:
                    focal_length = (target_w / 2.0) / math.tan(math.radians(self.fov) / 2.0)
                    center = (target_w / 2.0, target_h / 2.0)
                    cached_camera_matrix = np.array(
                        [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype="double"
                    )
                    cached_dims = (target_w, target_h)
                
                timestamp_ms = int(time.time() * 1000)

                # --- ROI 处理逻辑 ---
                roi_info = None # (roi_x, roi_y, roi_w, roi_h) in processed_rgb pixel coords
                processed_rgb = None
                
                # 检查是否需要进行全图扫描 (ROI 不存在，或者间隔达到)
                should_process_hand = True
                
                # 1. 尝试 ROI 模式
                if tracker.roi:
                    # ROI 模式：仅获取 ROI 区域 (注意这里使用 BGR)
                    # 先从原始 BGR 帧裁剪
                    cropped_roi, roi_rect = GlobalImagePreprocessor.crop_by_normalized_roi(frame, tracker.roi)
                    if cropped_roi is not None:
                        # 降分辨率 (ROI 缩放) - BGR
                        resized_roi = GlobalImagePreprocessor.resize_image(cropped_roi, scale_factor=settings.PREPROCESS_ROI_SCALE_FACTOR)
                        # 转换 RGB (仅 ROI 区域)
                        processed_rgb = GlobalImagePreprocessor.to_rgb(resized_roi)
                        roi_info = roi_rect
                    else:
                        # ROI 无效，回退到全图
                        tracker.roi = None
                        tracker.roi_miss_count = 0
                
                # 2. 准备全图图像 (如果手部全图扫描)
                need_full_frame = (not tracker.roi and frame_id % settings.FULL_SCAN_INTERVAL == 0)
                processed_rgb_full = None
                
                if need_full_frame:
                    resized_bgr = GlobalImagePreprocessor.resize_image(frame, target_size=(target_w, target_h))
                    processed_rgb_full = GlobalImagePreprocessor.to_rgb(resized_bgr)
                    # 全图也进行模糊处理
                    processed_rgb_full = GlobalImagePreprocessor.apply_gaussian_blur(processed_rgb_full, kernel_size=settings.PREPROCESS_GAUSSIAN_KERNEL_SIZE, sigma=settings.PREPROCESS_GAUSSIAN_SIGMA)
                
                # 3. 如果没有 ROI，使用全图作为手部检测输入
                if not tracker.roi:
                    if processed_rgb_full is not None:
                        processed_rgb = processed_rgb_full
                    else:
                        should_process_hand = False
                
                mapped_landmarks_list = []
                
                if should_process_hand and processed_rgb is not None:
                    # 3. 高斯模糊 (对 ROI 或 全图 都应用)
                    if processed_rgb is not processed_rgb_full:
                        processed_rgb = GlobalImagePreprocessor.apply_gaussian_blur(processed_rgb, kernel_size=settings.PREPROCESS_GAUSSIAN_KERNEL_SIZE, sigma=settings.PREPROCESS_GAUSSIAN_SIGMA)
                    
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=processed_rgb)
                    
                    # MediaPipe 处理
                    detection_result = tracker.process(mp_image, timestamp_ms)
                    
                    # --- 坐标映射回全图 ---
                    if detection_result.hand_landmarks:
                        tracker.roi_miss_count = 0 # 重置丢失计数
                        
                        for landmarks in detection_result.hand_landmarks:
                            mapped_landmarks = []
                            for lm in landmarks:
                                # 1. 还原到当前图像 (processed_rgb) 像素坐标
                                h_curr, w_curr = processed_rgb.shape[:2]
                                px = lm.x * w_curr
                                py = lm.y * h_curr
                                
                                if roi_info:
                                    # ROI 模式：还原 ROI 缩放和偏移
                                    px = px / settings.PREPROCESS_ROI_SCALE_FACTOR
                                    py = py / settings.PREPROCESS_ROI_SCALE_FACTOR
                                    
                                    roi_x, roi_y, _, _ = roi_info
                                    px += roi_x
                                    py += roi_y
                                    
                                    # 归一化回 720p 目标分辨率
                                    final_x = (px / w) * target_w
                                    final_y = (py / h) * target_h
                                else:
                                    final_x = px
                                    final_y = py
                                
                                norm_x = final_x / target_w
                                norm_y = final_y / target_h
                                
                                mapped_landmarks.append(LandmarkLite(norm_x, norm_y, lm.z))
                            mapped_landmarks_list.append(mapped_landmarks)
                        
                        # 更新 ROI
                        next_roi = tracker.calculate_roi(mapped_landmarks_list)
                        if next_roi:
                            tracker.roi = next_roi
                    else:
                        tracker.roi_miss_count += 1
                        mapped_landmarks_list = []
                        if tracker.roi_miss_count > tracker.MAX_ROI_MISS_COUNT:
                            tracker.roi = None 
                else:
                    pass
                
                # --- 过滤重叠手部 (使用映射后的坐标) ---
                filtered_landmarks = []
                filtered_handedness = []
                
                if should_process_hand and 'detection_result' in locals():
                    filtered_landmarks, filtered_handedness = tracker.filter_overlapping_hands(
                        mapped_landmarks_list, 
                        detection_result.handedness
                    )
                
                result_lite = HandDetectionResultLite(filtered_landmarks, filtered_handedness)
                
                # 计算空间位置并找到最近的手
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
                        
                        # 使用 HandTracker 内部的 filters 进行计算
                        x, y, z, w_norm, yaw, pitch, motion_score, grip_factor, depth_details = tracker.calculate_hand_pos(
                            landmarks, target_w, target_h, aspect_ratio, 
                            timestamp=timestamp_ms / 1000.0,
                            camera_matrix=cached_camera_matrix,
                            hand_label=label,
                            frame_id=frame_id
                        )
                        
                        if x is not None:
                            # 检测 Pinch
                            is_pinching, px, py, pz, pinch_cx, pinch_cy = tracker.detect_pinch(landmarks, z, aspect_ratio)
                            
                            hand_info = {
                                'id': idx,
                                'label': label,
                                'x': x,
                                'y': y,
                                'z': z,
                                'yaw': yaw,
                                'pitch': pitch,
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

                # 将结果放入输出队列
                if self.output_queue.full():
                    try:
                        self.output_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.output_queue.put({
                    'frame_id': frame_id,
                    'hand_result': result_lite,
                    'timestamp': timestamp_ms,
                    'closest_hand': closest_hand_info,
                    'hands_pos': hands_pos
                })
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing Error in Hand Process: {e}")

        # 清理
        tracker.close()
        for mgr in self.shm_managers:
            try:
                mgr.close()
            except:
                pass
