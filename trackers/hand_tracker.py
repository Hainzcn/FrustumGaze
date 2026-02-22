
import multiprocessing
import queue
import time
import cv2
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from modules.shared_mem import get_shared_array
from utils.math_utils import OneEuroFilter, Simple3DKalmanFilter
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
    def __init__(self, input_queue, output_queue, stop_event, shm_name, frame_shape, fov=60.0):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.shm_name = shm_name
        self.frame_shape = frame_shape
        self.fov = fov
        self.daemon = True

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
        PINCH_THRESHOLD_M = 0.02 
        
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
            
            # 计算捏起点空间坐标
            px = z_depth * (cx - 0.5) * 2.0 * tan_half_fov
            py = z_depth * (cy - 0.5) * (1.0 / aspect_ratio) * 2.0 * tan_half_fov
            pz = z_depth # 简化
            
            return True, px, py, pz, cx, cy
            
        return False, 0.0, 0.0, 0.0, 0.0, 0.0

    def _calculate_hand_pos(self, landmarks, aspect_ratio, w_norm_filter=None, pos_filter=None, timestamp=None):
        """
        计算手部空间位置 (Camera Space)
        假设: 手掌宽度 (Index MCP 5 -> Pinky MCP 17) 约为 8cm (0.08m)
        """
        HAND_WIDTH_REAL = 0.08  # meters
        
        # 获取关键点
        p5 = landmarks[5]  # INDEX_FINGER_MCP
        p17 = landmarks[17] # PINKY_MCP
        
        # 计算图像平面上的归一化距离 (仅 x, y)
        dx = p5.x - p17.x
        dy = p5.y - p17.y
        w_norm = math.sqrt(dx*dx + dy*dy)
        
        if w_norm < 1e-6:
            return None, None, None, None

        # --- 应用 OneEuroFilter 滤波 (对 w_norm) ---
        if w_norm_filter and timestamp is not None:
            w_norm = w_norm_filter.filter(w_norm, timestamp)

        # 计算 Z (深度)
        # Z = W_real / (2 * w_norm * tan(fov/2))
        # 注意: 这里假设 fov 是水平视场角
        tan_half_fov = math.tan(math.radians(self.fov) / 2.0)
        z = HAND_WIDTH_REAL / (2.0 * w_norm * tan_half_fov)
        
        # 计算 X, Y
        # 使用手掌中心 (例如 Index MCP 和 Pinky MCP 的中点，或者 WRIST)
        # 这里使用 5 和 17 的中点作为手掌中心
        cx = (p5.x + p17.x) / 2.0
        cy = (p5.y + p17.y) / 2.0
        
        # X = Z * (cx - 0.5) * 2 * tan(fov/2)
        x = z * (cx - 0.5) * 2.0 * tan_half_fov
        
        # Y = Z * (cy - 0.5) * 2 * tan(fov_v/2)
        # 考虑到 aspect_ratio = W / H
        # tan(fov_v/2) = tan(fov_h/2) / aspect_ratio (近似，或严格推导)
        # 简单推导: Y / Z = (y_pixel - H/2) / f
        # f = (W/2) / tan_half_fov
        # Y = Z * (cy - 0.5) * H / f
        #   = Z * (cy - 0.5) * H * 2 * tan_half_fov / W
        #   = Z * (cy - 0.5) * (1/aspect_ratio) * 2 * tan_half_fov
        y = z * (cy - 0.5) * (1.0 / aspect_ratio) * 2.0 * tan_half_fov
        
        # 坐标系: 
        # X: 右为正
        # Y: 下为正 (OpenCV 默认) -> 也可以转为 上为正 (-y)
        # Z: 前为正

        # --- 应用 SimpleKalmanFilter 滤波 (对 X, Y, Z) ---
        if pos_filter:
            x, y, z = pos_filter.update(x, y, z)
        
        return x, y, z, w_norm

    def run(self):
        # --- 在子进程中初始化资源 ---
        
        # 1. 连接共享内存
        try:
            shm_manager, shm_array = get_shared_array(self.shm_name, self.frame_shape)
        except Exception as e:
            print(f"HandProcessorProcess: Failed to connect to shared memory: {e}")
            return

        # 2. 初始化 MediaPipe Hands (Tasks API)
        try:
            base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
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
                    min_cutoff=settings.HAND_ONE_EURO_MIN_CUTOFF, 
                    beta=settings.HAND_ONE_EURO_BETA,
                    d_cutoff=settings.HAND_ONE_EURO_D_CUTOFF
                ),
                'pos': Simple3DKalmanFilter(
                    process_noise=settings.HAND_KALMAN_PROCESS_NOISE, 
                    measurement_noise=settings.HAND_KALMAN_MEASUREMENT_NOISE
                )
            },
            'Right': {
                'w_norm': OneEuroFilter(
                    min_cutoff=settings.HAND_ONE_EURO_MIN_CUTOFF, 
                    beta=settings.HAND_ONE_EURO_BETA,
                    d_cutoff=settings.HAND_ONE_EURO_D_CUTOFF
                ),
                'pos': Simple3DKalmanFilter(
                    process_noise=settings.HAND_KALMAN_PROCESS_NOISE, 
                    measurement_noise=settings.HAND_KALMAN_MEASUREMENT_NOISE
                )
            }
        }

        while not self.stop_event.is_set():
            try:
                # 阻塞等待任务
                task = self.input_queue.get(timeout=0.01)
                frame_id = task['frame_id']
                
                # 从共享内存复制图像数据
                frame = shm_array.copy()
                
                # 降分辨率处理
                h, w = frame.shape[:2]
                aspect_ratio = w / float(h)
                target_h = 720
                scale = target_h / float(h)
                target_w = int(w * scale)
                
                processed_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_rgb = cv2.resize(processed_rgb, (target_w, target_h))
                
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=processed_rgb)
                timestamp_ms = int(time.time() * 1000)
                
                # MediaPipe 处理
                detection_result = detector.detect_for_video(mp_image, timestamp_ms)
                
                # --- 过滤重叠手部 (新增) ---
                filtered_landmarks, filtered_handedness = self._filter_overlapping_hands(
                    detection_result.hand_landmarks, 
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
                        label = "Unknown"
                        
                        if result_lite.multi_handedness and idx < len(result_lite.multi_handedness):
                            # handedness[0] is the category with highest score
                            categories = result_lite.multi_handedness[idx]
                            if categories:
                                label = categories[0]['label'] # "Left" or "Right"
                                if label in self.hand_filters:
                                    w_norm_filter = self.hand_filters[label]['w_norm']
                                    pos_filter = self.hand_filters[label]['pos']
                        
                        x, y, z, w_norm = self._calculate_hand_pos(
                            landmarks, aspect_ratio, 
                            w_norm_filter=w_norm_filter, 
                            pos_filter=pos_filter, 
                            timestamp=timestamp_ms / 1000.0
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
        shm_manager.close()
