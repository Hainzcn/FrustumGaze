
import multiprocessing
import queue
import time
import cv2
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from modules.shared_mem import get_shared_array
from utils.math_utils import OneEuroFilter, Simple3DKalmanFilter
from utils.image_utils import GlobalImagePreprocessor
from config import settings

# 定义简单的 Landmark 类以便于 Pickle
class LandmarkLite:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

# 定义简单的 Result 类以便于 Pickle
class HandDetectionResultLite:
    def __init__(self, hand_landmarks_list, handedness_list=None):
        self.multi_hand_landmarks = []
        self.multi_handedness = []
        
        if hand_landmarks_list:
            for landmarks in hand_landmarks_list:
                simple_landmarks = []
                for lm in landmarks:
                    simple_landmarks.append(LandmarkLite(lm.x, lm.y, lm.z))
                self.multi_hand_landmarks.append(simple_landmarks)
        if handedness_list:
            # MediaPipe's handedness is a list of lists of categories
            for handedness in handedness_list:
                simple_handedness = []
                for category in handedness:
                    # category has index, score, display_name, category_name
                    simple_handedness.append({
                        'score': category.score, 
                        'label': category.category_name,
                        'index': category.index
                    })
                self.multi_handedness.append(simple_handedness)

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
        # ROI 状态: (x_min, y_min, x_max, y_max) 归一化坐标 (0-1)
        self.roi = None
        self.roi_miss_count = 0
        self.MAX_ROI_MISS_COUNT = 30 # 连续多少帧没检测到手重置 ROI

    def _calculate_roi(self, landmarks_list, padding_factor=0.5):
        """
        根据当前检测到的手部计算下一帧的 ROI
        返回: (x_min, y_min, x_max, y_max) 归一化坐标
        """
        if not landmarks_list:
            return None
            
        all_x = []
        all_y = []
        
        for landmarks in landmarks_list:
            for lm in landmarks:
                all_x.append(lm.x)
                all_y.append(lm.y)
                
        if not all_x:
            return None
            
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        
        w = x_max - x_min
        h = y_max - y_min
        
        # 扩展边界
        pad_x = w * padding_factor
        pad_y = h * padding_factor
        
        # 确保 ROI 不过小
        min_size = 0.2 # 最小占画面 20% ? 不，太大了。如果不扩展可能会太小。
        # 还是只做 padding 吧。
        
        roi_x_min = max(0.0, x_min - pad_x)
        roi_y_min = max(0.0, y_min - pad_y)
        roi_x_max = min(1.0, x_max + pad_x)
        roi_y_max = min(1.0, y_max + pad_y)
        
        return (roi_x_min, roi_y_min, roi_x_max, roi_y_max)

    def _calculate_bbox(self, landmarks):
        """计算手部边界框 (normalized coordinates)"""
        x_min = min([lm.x for lm in landmarks])
        y_min = min([lm.y for lm in landmarks])
        x_max = max([lm.x for lm in landmarks])
        y_max = max([lm.y for lm in landmarks])
        return (x_min, y_min, x_max, y_max)

    def _calculate_iou(self, box1, box2):
        """计算两个边界框的 IoU 和包含率"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # 计算交集区域
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0, 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        # 计算并集区域
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return 0.0, 0.0
            
        iou = inter_area / union_area
        
        # 计算包含率 (Intersection over Self Area)
        # 只要有一个框被另一个框严重包含，就返回较高的包含率
        overlap_1 = inter_area / area1 if area1 > 0 else 0
        overlap_2 = inter_area / area2 if area2 > 0 else 0
        max_overlap = max(overlap_1, overlap_2)

        return iou, max_overlap

    def _filter_overlapping_hands(self, landmarks_list, handedness_list, iou_threshold=0.5, overlap_threshold=0.7):
        """
        过滤重叠严重的手部检测结果
        如果两只手的 IoU > threshold 或 包含率 > overlap_threshold，保留置信度更高的那只
        """
        if not landmarks_list or len(landmarks_list) < 2:
            return landmarks_list, handedness_list

        num_hands = len(landmarks_list)
        keep_indices = set(range(num_hands))
        
        # 计算所有手的边界框
        bboxes = [self._calculate_bbox(lm) for lm in landmarks_list]
        
        # 获取每只手的最高置信度
        scores = []
        for h_list in handedness_list:
            # 取该手最高 score 的 category
            max_score = 0.0
            if h_list:
                max_score = max([cat.score for cat in h_list])
            scores.append(max_score)

        # 两两比较
        sorted_indices = sorted(range(num_hands), key=lambda k: scores[k], reverse=True)
        
        final_indices = []
        
        for i in sorted_indices:
            if i not in keep_indices:
                continue
            
            is_kept = True
            for j in final_indices:
                # 检查与已保留的手是否重叠
                iou, max_overlap = self._calculate_iou(bboxes[i], bboxes[j])
                
                # 如果 IoU 过高 或者 存在严重的包含关系 (大框包小框)
                if iou > iou_threshold or max_overlap > overlap_threshold:
                    is_kept = False # 与更高置信度的手冲突，丢弃 i
                    break
            
            if is_kept:
                final_indices.append(i)
        
        # 根据 final_indices 重建列表
        filtered_landmarks = [landmarks_list[i] for i in final_indices]
        filtered_handedness = [handedness_list[i] for i in final_indices]
        
        return filtered_landmarks, filtered_handedness

    def _detect_pinch(self, landmarks, z_depth, aspect_ratio):
        """
        检测是否捏起 (拇指与其他手指)
        返回: (is_pinching, pinch_x, pinch_y, pinch_z)
        """
        # 关键点索引
        THUMB_TIP = 4
        INDEX_TIP = 8
        MIDDLE_TIP = 12
        RING_TIP = 16
        PINKY_TIP = 20
        
        TIPS = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
        
        thumb = landmarks[THUMB_TIP]
        
        # 阈值设定: 2cm (0.02m)
        PINCH_THRESHOLD_M = settings.PINCH_THRESHOLD_M
        
        # 转换因子
        tan_half_fov = math.tan(math.radians(self.fov) / 2.0)
        
        pinching_fingers = []
        
        for tip_idx in TIPS:
            finger = landmarks[tip_idx]
            dx = thumb.x - finger.x
            dy = thumb.y - finger.y
            
            # 近似实际距离
            dx_m = dx * z_depth * 2.0 * tan_half_fov
            dy_m = dy * z_depth * (1.0 / aspect_ratio) * 2.0 * tan_half_fov
            
            dist_m = math.sqrt(dx_m*dx_m + dy_m*dy_m)
            
            if dist_m < PINCH_THRESHOLD_M:
                pinching_fingers.append(finger)
        
        # 只要有一根手指与拇指接触，就算捏起 (即 >= 2根手指参与)
        if len(pinching_fingers) > 0:
            # 计算捏起中心
            # 取拇指和所有捏起指尖的平均
            sum_x = thumb.x
            sum_y = thumb.y
            count = 1
            
            for f in pinching_fingers:
                sum_x += f.x
                sum_y += f.y
                count += 1
                
            cx = sum_x / count
            cy = sum_y / count
            
            return True, 0.0, 0.0, 0.0, cx, cy
            
        return False, 0.0, 0.0, 0.0, 0.0, 0.0

    def _calculate_hand_pos(self, landmarks, frame_width, frame_height, aspect_ratio, w_norm_filter=None, pos_filter=None, one_euro_filter_dict=None, timestamp=None, camera_matrix=None):
        """
        计算手部空间位置 (Camera Space)
        使用 PnP 解算
        关键点索引: 0 (Wrist), 5 (Index MCP), 9 (Middle MCP), 13 (Ring MCP), 17 (Pinky MCP)
        模型设定 (以 5-17 连线中点为原点, 使得 PnP 结果直接反映手掌中心位置):
        - 距离: 0-5=10cm, 5-17=6cm, 0-17=8cm (默认)
        - 直角三角形, 直角在 17
        - P5:  (0.0, -0.03, 0.0)
        - P9:  (0.0, -0.01, 0.0)
        - P13: (0.0,  0.01, 0.0)
        - P17: (0.0,  0.03, 0.0)
        - P0:  (0.08, 0.03, 0.0)
        """
        
        # 3D Model Points (Meters)
        # 根据配置的手掌宽度进行缩放
        # 默认宽度 6.0cm
        scale = settings.HAND_PALM_WIDTH_CM / 6.0
        
        # 为了避免 3 点共线导致 SOLVEPNP_IPPE 失败，对中间手指的 X 坐标进行微调
        # 模拟指关节的自然弧度
        model_points = np.array([
            (0.08 * scale, 0.03 * scale, 0.0),     # 0: Wrist
            (0.0, -0.03 * scale, 0.0),             # 5: Index MCP
            (-0.01 * scale, -0.01 * scale, 0.0),   # 9: Middle MCP (稍向前突出)
            (-0.005 * scale,  0.01 * scale, 0.0),  # 13: Ring MCP (稍向前突出)
            (0.0,  0.03 * scale, 0.0)              # 17: Pinky MCP
        ], dtype="double")
        
        # --- 应用 OneEuroFilter 滤波 (对关键点) ---
        p0_x, p0_y = landmarks[0].x, landmarks[0].y
        p5_x, p5_y = landmarks[5].x, landmarks[5].y
        p9_x, p9_y = landmarks[9].x, landmarks[9].y
        p13_x, p13_y = landmarks[13].x, landmarks[13].y
        p17_x, p17_y = landmarks[17].x, landmarks[17].y
        
        if one_euro_filter_dict is not None and timestamp is not None:
            def get_filtered_val(name, val):
                if name not in one_euro_filter_dict:
                    one_euro_filter_dict[name] = OneEuroFilter(
                        min_cutoff=settings.HAND_POS_ONE_EURO_MIN_CUTOFF, 
                        beta=settings.HAND_POS_ONE_EURO_BETA,
                        d_cutoff=settings.HAND_POS_ONE_EURO_D_CUTOFF
                    )
                return one_euro_filter_dict[name].filter(val, timestamp)
            
            p0_x = get_filtered_val('p0_x', p0_x)
            p0_y = get_filtered_val('p0_y', p0_y)
            p5_x = get_filtered_val('p5_x', p5_x)
            p5_y = get_filtered_val('p5_y', p5_y)
            p9_x = get_filtered_val('p9_x', p9_x)
            p9_y = get_filtered_val('p9_y', p9_y)
            p13_x = get_filtered_val('p13_x', p13_x)
            p13_y = get_filtered_val('p13_y', p13_y)
            p17_x = get_filtered_val('p17_x', p17_x)
            p17_y = get_filtered_val('p17_y', p17_y)

        # 2D Image Points
        image_points = np.array([
            (p0_x * frame_width, p0_y * frame_height),   # 0
            (p5_x * frame_width, p5_y * frame_height),   # 5
            (p9_x * frame_width, p9_y * frame_height),   # 9
            (p13_x * frame_width, p13_y * frame_height), # 13
            (p17_x * frame_width, p17_y * frame_height)  # 17
        ], dtype="double")
        
        # Camera Matrix
        if camera_matrix is None:
            # fx = fy = (w / 2) / tan(fov / 2)
            focal_length = (frame_width / 2.0) / math.tan(math.radians(self.fov) / 2.0)
            center = (frame_width / 2.0, frame_height / 2.0)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype="double"
            )
        dist_coeffs = np.zeros((4, 1)) # Assuming no distortion
        
        # 使用 IPPE 方法解决 PnP (需要 4 个共面点)
        # 选取 0, 5, 13, 17 作为 4 个关键点
        model_points_4 = model_points[[0, 1, 3, 4]]
        image_points_4 = image_points[[0, 1, 3, 4]]
        
        rvecs, tvecs = [], []
        success = False
        
        try:
            # SOLVEPNP_IPPE 适用于 4 个共面点，返回所有可能的解 (通常是 2 个)
            # 在部分 OpenCV 版本中可能返回 4 个值 (n, rvecs, tvecs, err)
            pnp_result = cv2.solvePnPGeneric(
                model_points_4, image_points_4, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE
            )
            
            if len(pnp_result) == 3:
                n_solutions, rvecs, tvecs = pnp_result
            elif len(pnp_result) == 4:
                n_solutions, rvecs, tvecs, _ = pnp_result
            else:
                success = False
                n_solutions = 0

            success = n_solutions > 0
        except Exception:
            success = False

        # Fallback: 如果 IPPE 失败，尝试 ITERATIVE (使用所有 5 个点)
        if not success:
            try:
                success_iter, rvec_iter, tvec_iter = cv2.solvePnP(
                    model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
                )
                if success_iter:
                    rvecs = [rvec_iter]
                    tvecs = [tvec_iter]
                    success = True
            except Exception:
                success = False

        if not success:
            return None, None, None, None, 0.0

        # --- 解的筛选 (Disambiguation) ---
        # 计算观测到的手掌法向量 (Camera Space)
        # 使用 landmarks 的 (x, y, z) 构建向量 P5-P0 和 P17-P0
        # 注意: landmarks.z 是相对于 wrist 的深度，比例尺大致与 x 相同
        # 需要考虑 aspect ratio 将 y 转换到与 x, z 相同的比例空间
        
        def get_vec(lm_to, lm_from):
            return np.array([
                lm_to.x - lm_from.x,
                (lm_to.y - lm_from.y) * (frame_height / frame_width), # Normalize y scale to x
                lm_to.z - lm_from.z
            ])

        v1_obs = get_vec(landmarks[5], landmarks[0])  # P0 -> P5
        v2_obs = get_vec(landmarks[17], landmarks[0]) # P0 -> P17
        normal_obs = np.cross(v1_obs, v2_obs)
        norm_obs_val = np.linalg.norm(normal_obs)
        if norm_obs_val > 1e-6:
            normal_obs /= norm_obs_val
        
        # 计算模型法向量 (Model Space)
        v1_model = model_points[1] - model_points[0] # P0 -> P5
        v2_model = model_points[4] - model_points[0] # P0 -> P17
        normal_model = np.cross(v1_model, v2_model)
        norm_model_val = np.linalg.norm(normal_model)
        if norm_model_val > 1e-6:
            normal_model /= norm_model_val
            
        best_rvec = None
        best_tvec = None
        max_dot = -100.0
        
        for i in range(len(rvecs)):
            rvec = rvecs[i]
            tvec = tvecs[i]
            
            # 将模型法向量变换到相机空间: N_cam = R * N_model
            R, _ = cv2.Rodrigues(rvec)
            normal_cam = np.dot(R, normal_model)
            
            # 计算与观测法向量的点积
            dot_prod = np.dot(normal_cam, normal_obs)
            
            if dot_prod > max_dot:
                max_dot = dot_prod
                best_rvec = rvec
                best_tvec = tvec
        
        if best_rvec is None:
            return None, None, None, None, 0.0
            
        rotation_vector = best_rvec
        translation_vector = best_tvec

        x = translation_vector[0][0]
        y = translation_vector[1][0]
        z = translation_vector[2][0]

        # 计算 Yaw 角 (围绕 Y 轴旋转)
        # rotation_vector 是旋转向量，需要转换为旋转矩阵
        R, _ = cv2.Rodrigues(rotation_vector)
        # Yaw 计算通常依赖于旋转矩阵的具体定义。假设标准相机坐标系:
        
        # 简化计算 (假设主要关注水平旋转)
        # Yaw = atan2(R[0,2], R[2,2]) (视定义而定)
        # 这里使用通用的 Euler Angles 计算
        
        yaw = math.atan2(-R[2,0], math.sqrt(R[2,1]**2 + R[2,2]**2))
        yaw_deg = math.degrees(yaw)
        
        # --- 应用 OneEuroFilter 滤波 (对 Yaw 角) ---
        if one_euro_filter_dict is not None and timestamp is not None:
            if 'yaw' not in one_euro_filter_dict:
                one_euro_filter_dict['yaw'] = OneEuroFilter(
                    min_cutoff=settings.HAND_YAW_ONE_EURO_MIN_CUTOFF, 
                    beta=settings.HAND_YAW_ONE_EURO_BETA,
                    d_cutoff=settings.HAND_YAW_ONE_EURO_D_CUTOFF
                )
            yaw_deg = one_euro_filter_dict['yaw'].filter(yaw_deg, timestamp)
        
        # 为了兼容旧逻辑，计算 w_norm 作为某种置信度或调试信息
        dx = landmarks[5].x - landmarks[17].x
        dy = (landmarks[5].y - landmarks[17].y) * (1.0 / aspect_ratio)
        w_norm = math.sqrt(dx*dx + dy*dy)
        
        # --- 应用 SimpleKalmanFilter 滤波 (对 X, Y, Z) ---
        if pos_filter:
            x, y, z = pos_filter.update(x, y, z)
        
        return x, y, z, w_norm, yaw_deg

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

        # 2. 初始化 MediaPipe Hands (Tasks API)
        try:
            base_options = python.BaseOptions(model_asset_path=settings.HAND_LANDMARKER_TASK_PATH)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=2,
                min_hand_detection_confidence=settings.HAND_MIN_DETECTION_CONFIDENCE,
                min_hand_presence_confidence=settings.HAND_MIN_PRESENCE_CONFIDENCE,
                min_tracking_confidence=settings.HAND_MIN_TRACKING_CONFIDENCE,
                running_mode=vision.RunningMode.VIDEO)
            detector = vision.HandLandmarker.create_from_options(options)
        except Exception as e:
            print(f"HandProcessorProcess: Failed to init MediaPipe: {e}")
            return
        
        print(f"HandProcessorProcess: Started and Ready. FOV={self.fov}")

        # 初始化滤波器
        self.hand_filters = {
            'Left': {
                'w_norm': OneEuroFilter(
                    min_cutoff=settings.HAND_DIST_ONE_EURO_MIN_CUTOFF, 
                    beta=settings.HAND_DIST_ONE_EURO_BETA,
                    d_cutoff=settings.HAND_DIST_ONE_EURO_D_CUTOFF
                ),
                'pos': Simple3DKalmanFilter(
                    process_noise=settings.HAND_KALMAN_PROCESS_NOISE, 
                    measurement_noise=settings.HAND_KALMAN_MEASUREMENT_NOISE
                )
            },
            'Right': {
                'w_norm': OneEuroFilter(
                    min_cutoff=settings.HAND_DIST_ONE_EURO_MIN_CUTOFF, 
                    beta=settings.HAND_DIST_ONE_EURO_BETA,
                    d_cutoff=settings.HAND_DIST_ONE_EURO_D_CUTOFF
                ),
                'pos': Simple3DKalmanFilter(
                    process_noise=settings.HAND_KALMAN_PROCESS_NOISE, 
                    measurement_noise=settings.HAND_KALMAN_MEASUREMENT_NOISE
                )
            }
        }

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
                # 优化：直接使用共享内存，避免全量拷贝
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
                
                # processed_rgb = GlobalImagePreprocessor.to_rgb(frame) # 移除：延迟转换
                
                timestamp_ms = int(time.time() * 1000)

                # --- ROI 处理逻辑 ---
                roi_info = None # (roi_x, roi_y, roi_w, roi_h) in processed_rgb pixel coords
                processed_rgb = None
                
                # 检查是否需要进行全图扫描 (ROI 不存在，或者间隔达到)
                should_process_hand = True
                
                # 1. 尝试 ROI 模式
                if self.roi:
                    # ROI 模式：仅获取 ROI 区域 (注意这里使用 BGR)
                    # 先从原始 BGR 帧裁剪
                    cropped_roi, roi_rect = GlobalImagePreprocessor.crop_by_normalized_roi(frame, self.roi)
                    if cropped_roi is not None:
                        # 降分辨率 (ROI 缩放) - BGR
                        resized_roi = GlobalImagePreprocessor.resize_image(cropped_roi, scale_factor=settings.PREPROCESS_ROI_SCALE_FACTOR)
                        # 转换 RGB (仅 ROI 区域)
                        processed_rgb = GlobalImagePreprocessor.to_rgb(resized_roi)
                        roi_info = roi_rect
                    else:
                        # ROI 无效，回退到全图
                        self.roi = None
                        self.roi_miss_count = 0
                
                # 2. 准备全图图像 (如果手部全图扫描)
                need_full_frame = (not self.roi and frame_id % settings.FULL_SCAN_INTERVAL == 0)
                processed_rgb_full = None
                
                if need_full_frame:
                    resized_bgr = GlobalImagePreprocessor.resize_image(frame, target_size=(target_w, target_h))
                    processed_rgb_full = GlobalImagePreprocessor.to_rgb(resized_bgr)
                    # 全图也进行模糊处理
                    processed_rgb_full = GlobalImagePreprocessor.apply_gaussian_blur(processed_rgb_full, kernel_size=settings.PREPROCESS_GAUSSIAN_KERNEL_SIZE, sigma=settings.PREPROCESS_GAUSSIAN_SIGMA)
                
                # 3. 如果没有 ROI，使用全图作为手部检测输入
                if not self.roi:
                    if processed_rgb_full is not None:
                        processed_rgb = processed_rgb_full
                    else:
                        should_process_hand = False
                
                mapped_landmarks_list = []
                
                if should_process_hand and processed_rgb is not None:
                    # 3. 高斯模糊 (对 ROI 或 全图 都应用)
                    # 如果是 ROI 图像，需要单独模糊 (全图已经模糊过了)
                    if processed_rgb is not processed_rgb_full:
                        processed_rgb = GlobalImagePreprocessor.apply_gaussian_blur(processed_rgb, kernel_size=settings.PREPROCESS_GAUSSIAN_KERNEL_SIZE, sigma=settings.PREPROCESS_GAUSSIAN_SIGMA)
                    
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=processed_rgb)
                    
                    # MediaPipe 处理
                    detection_result = detector.detect_for_video(mp_image, timestamp_ms)
                    
                    # --- 坐标映射回全图 ---
                    if detection_result.hand_landmarks:
                        self.roi_miss_count = 0 # 重置丢失计数
                        
                        for landmarks in detection_result.hand_landmarks:
                            mapped_landmarks = []
                            for lm in landmarks:
                                # 1. 还原到当前图像 (processed_rgb) 像素坐标
                                h_curr, w_curr = processed_rgb.shape[:2]
                                px = lm.x * w_curr
                                py = lm.y * h_curr
                                
                                if roi_info:
                                    # ROI 模式：还原 ROI 缩放和偏移
                                    # 2. 还原 ROI Resize
                                    px = px / settings.PREPROCESS_ROI_SCALE_FACTOR
                                    py = py / settings.PREPROCESS_ROI_SCALE_FACTOR
                                    
                                    # 3. 还原 ROI 偏移 (基于原图)
                                    roi_x, roi_y, _, _ = roi_info
                                    px += roi_x
                                    py += roi_y
                                    
                                    # 4. 归一化回 720p 目标分辨率 (为了与 PnP 兼容)
                                    final_x = (px / w) * target_w
                                    final_y = (py / h) * target_h
                                    
                                else:
                                    # 全图模式：输入已经是 resize 到 target_w 的图像
                                    final_x = px
                                    final_y = py
                                
                                # 归一化坐标用于 ROI 更新
                                norm_x = final_x / target_w
                                norm_y = final_y / target_h
                                
                                mapped_landmarks.append(LandmarkLite(norm_x, norm_y, lm.z))
                            mapped_landmarks_list.append(mapped_landmarks)
                        
                        # 更新 ROI
                        next_roi = self._calculate_roi(mapped_landmarks_list)
                        if next_roi:
                            self.roi = next_roi
                    else:
                        self.roi_miss_count += 1
                        mapped_landmarks_list = [] # 空列表
                        if self.roi_miss_count > self.MAX_ROI_MISS_COUNT:
                            self.roi = None # 丢失太久，重置为全图扫描
                else:
                    # 如果跳过处理 (全图模式下的非扫描帧)，返回空结果或者沿用上一帧结果？
                    # 这里返回空结果，让主线程处理
                    pass
                
                # 替换原始结果中的 landmarks 以供后续逻辑使用
                # 注意：detect_for_video 返回的是 immutable 对象结构，无法直接修改内部属性
                # 但后续逻辑使用的是 filtered_landmarks，我们可以在这里拦截并替换
                
                # --- 过滤重叠手部 (使用映射后的坐标) ---
                # 注意：如果 should_process 为 False，detection_result 未定义，需要处理这种情况
                
                filtered_landmarks = []
                filtered_handedness = []
                
                if should_process_hand and 'detection_result' in locals():
                    filtered_landmarks, filtered_handedness = self._filter_overlapping_hands(
                        mapped_landmarks_list, 
                        detection_result.handedness
                    )
                
                result_lite = HandDetectionResultLite(filtered_landmarks, filtered_handedness)
                
                # 计算空间位置并找到最近的手
                closest_hand_info = None
                min_z = float('inf')
                
                # 存储所有手的空间位置，以便 Visualizer 使用
                hands_pos = []

                if result_lite.multi_hand_landmarks:
                    for idx, landmarks in enumerate(result_lite.multi_hand_landmarks):
                        # 获取滤波器
                        w_norm_filter = None
                        pos_filter = None
                        one_euro_filter_dict = None
                        label = "Unknown"
                        
                        if result_lite.multi_handedness and idx < len(result_lite.multi_handedness):
                            # handedness[0] is the category with highest score
                            categories = result_lite.multi_handedness[idx]
                            if categories:
                                label = categories[0]['label'] # "Left" or "Right"
                                if label in self.hand_filters:
                                    w_norm_filter = self.hand_filters[label]['w_norm']
                                    pos_filter = self.hand_filters[label]['pos']
                                    # 检查是否有 OneEuroFilter 字典用于关键点滤波
                                    if 'landmarks' not in self.hand_filters[label]:
                                        self.hand_filters[label]['landmarks'] = {}
                                    one_euro_filter_dict = self.hand_filters[label]['landmarks']
                        
                        x, y, z, w_norm, yaw = self._calculate_hand_pos(
                            landmarks, target_w, target_h, aspect_ratio, 
                            w_norm_filter=w_norm_filter, 
                            pos_filter=pos_filter, 
                            one_euro_filter_dict=one_euro_filter_dict,
                            timestamp=timestamp_ms / 1000.0,
                            camera_matrix=cached_camera_matrix
                        )
                        
                        if x is not None:
                            # 检测 Pinch
                            is_pinching, px, py, pz, pinch_cx, pinch_cy = self._detect_pinch(landmarks, z, aspect_ratio)
                            
                            hand_info = {
                                'id': idx,
                                'label': label,
                                'x': x,
                                'y': y,
                                'z': z,
                                'yaw': yaw,
                                'w_norm': w_norm,
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
        detector.close()
        for mgr in self.shm_managers:
            try:
                mgr.close()
            except:
                pass
