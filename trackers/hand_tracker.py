import time
import math
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from config import settings
from utils.math_utils import OneEuroFilter, Simple3DKalmanFilter
from .common import LandmarkLite

class HandDetectionResultLite:
    """
    轻量级手部检测结果包装类。
    用于简化 MediaPipe 的手部检测结果，便于跨进程或网络传输。
    """
    def __init__(self, hand_landmarks_list, handedness_list=None):
        self.multi_hand_landmarks = []
        self.multi_handedness = []
        
        if hand_landmarks_list:
            for landmarks in hand_landmarks_list:
                simple_landmarks = [LandmarkLite(lm.x, lm.y, lm.z) for lm in landmarks]
                self.multi_hand_landmarks.append(simple_landmarks)
        
        if handedness_list:
            for handedness in handedness_list:
                simple_handedness = [{
                    'score': category.score, 
                    'label': category.category_name,
                    'index': category.index
                } for category in handedness]
                self.multi_handedness.append(simple_handedness)

class HandTracker:
    """
    手部追踪类，集成 MediaPipe 手部关键点检测与 3D 空间位置估算。
    包含滤波、ROI 计算、重叠检测及基于几何关系的深度估计。
    """
    def __init__(self, fov=60.0):
        self.fov = fov
        self.detector = None
        self.roi = None
        self.roi_miss_count = 0
        self.MAX_ROI_MISS_COUNT = 30
        self._init_mediapipe()
        
        # 初始化手部过滤器状态
        self.hand_filters = {
            'Left': self._create_filter_state(),
            'Right': self._create_filter_state()
        }

    def _create_filter_state(self):
        """创建单个手部的滤波和状态跟踪字典"""
        hand_config = settings.FILTER_CONFIG['HAND']
        scale_config = hand_config['SCALE']
        pos_config = hand_config['POSITION']
        
        return {
            'w_norm': OneEuroFilter(
                min_cutoff=scale_config['min_cutoff'], 
                beta=scale_config['beta'],
                d_cutoff=scale_config['d_cutoff']
            ),
            'pos': Simple3DKalmanFilter(
                process_noise=pos_config['process_noise'], 
                measurement_noise=pos_config['measurement_noise']
            ),
            'depth_length_history': [],
            'depth_anchor': {'value': 0.0, 'frame_id': 0, 'timestamp': 0.0},
            'width_correction': {'value': 1.0, 'count': 0},
            'grip_state': {'value': 0.0},
            'pinch_debounce': {'raw': False, 'confirmed': False, 'count': 0},
            'landmarks': {} # 关键点的 OneEuroFilter 字典
        }

    def _init_mediapipe(self):
        """初始化 MediaPipe 手部检测器"""
        try:
            base_options = python.BaseOptions(model_asset_path=settings.HAND_LANDMARKER_TASK_PATH)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=2,
                min_hand_detection_confidence=settings.HAND_MIN_DETECTION_CONFIDENCE,
                min_hand_presence_confidence=settings.HAND_MIN_PRESENCE_CONFIDENCE,
                min_tracking_confidence=settings.HAND_MIN_TRACKING_CONFIDENCE,
                running_mode=vision.RunningMode.VIDEO)
            self.detector = vision.HandLandmarker.create_from_options(options)
            print(f"HandTracker: MediaPipe 初始化完成。")
        except Exception as e:
            print(f"HandTracker: MediaPipe 初始化失败: {e}")
            raise e

    def process(self, mp_image, timestamp_ms):
        """在视频帧上运行手部检测"""
        if not self.detector:
            return None
        return self.detector.detect_for_video(mp_image, timestamp_ms)

    def close(self):
        """释放 MediaPipe 资源"""
        if self.detector:
            self.detector.close()

    def get_filter_state(self, label):
        """获取指定手（左/右）的滤波状态"""
        return self.hand_filters.get(label)

    def calculate_roi(self, landmarks_list, padding_factor=0.5):
        """
        根据检测到的手部计算 ROI（感兴趣区域），用于加速下一帧检测。
        返回: (x_min, y_min, x_max, y_max) 归一化坐标
        """
        if not landmarks_list:
            return None
            
        all_x = [lm.x for landmarks in landmarks_list for lm in landmarks]
        all_y = [lm.y for landmarks in landmarks_list for lm in landmarks]
        
        if not all_x:
            return None
            
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        
        w, h = x_max - x_min, y_max - y_min
        pad_x, pad_y = w * padding_factor, h * padding_factor
        
        return (max(0.0, x_min - pad_x), max(0.0, y_min - pad_y), 
                min(1.0, x_max + pad_x), min(1.0, y_max + pad_y))

    def _calculate_bbox(self, landmarks):
        """计算手部关键点的归一化边界框"""
        x_min = min(lm.x for lm in landmarks)
        y_min = min(lm.y for lm in landmarks)
        x_max = max(lm.x for lm in landmarks)
        y_max = max(lm.y for lm in landmarks)
        return (x_min, y_min, x_max, y_max)

    def _calculate_iou(self, box1, box2):
        """计算两个边界框的 IoU 和重叠率"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        inter_x_min, inter_y_min = max(x1_min, x2_min), max(y1_min, y2_min)
        inter_x_max, inter_y_max = min(x1_max, x2_max), min(y1_max, y2_max)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0, 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return 0.0, 0.0
            
        iou = inter_area / union_area
        max_overlap = max(inter_area / area1 if area1 > 0 else 0, 
                          inter_area / area2 if area2 > 0 else 0)

        return iou, max_overlap

    def filter_overlapping_hands(self, landmarks_list, handedness_list, iou_threshold=0.5, overlap_threshold=0.7):
        """
        过滤由于检测误差导致的重叠手部检测结果。
        """
        if not landmarks_list or len(landmarks_list) < 2:
            return landmarks_list, handedness_list

        num_hands = len(landmarks_list)
        bboxes = [self._calculate_bbox(lm) for lm in landmarks_list]
        scores = [max(cat.score for cat in h_list) if h_list else 0.0 for h_list in handedness_list]

        # 优先保留置信度高的手
        sorted_indices = sorted(range(num_hands), key=lambda k: scores[k], reverse=True)
        keep_indices = []
        
        for i in sorted_indices:
            is_kept = True
            for j in keep_indices:
                iou, max_overlap = self._calculate_iou(bboxes[i], bboxes[j])
                if iou > iou_threshold or max_overlap > overlap_threshold:
                    is_kept = False 
                    break
            if is_kept:
                keep_indices.append(i)
        
        return ([landmarks_list[i] for i in keep_indices], 
                [handedness_list[i] for i in keep_indices])

    def _detect_pinch_raw(self, filtered_lms, z_depth, aspect_ratio):
        """
        使用滤波后的关键点检测拇指是否与其他手指尖捏合 (3D 距离判定)。
        返回 (raw_pinch, pinch_center_x, pinch_center_y)，坐标为归一化值。
        """
        THUMB_TIP = 4
        TIPS = [8, 12, 16, 20]

        thumb = filtered_lms[THUMB_TIP]
        tan_half_fov = math.tan(math.radians(self.fov) / 2.0)
        pinching_fingers = []

        for tip_idx in TIPS:
            if tip_idx not in filtered_lms:
                continue
            finger = filtered_lms[tip_idx]
            dx_m = (thumb.x - finger.x) * z_depth * 2.0 * tan_half_fov
            dy_m = (thumb.y - finger.y) * z_depth * (1.0 / aspect_ratio) * 2.0 * tan_half_fov
            dz_m = (thumb.z - finger.z) * z_depth * 2.0 * tan_half_fov

            if math.sqrt(dx_m**2 + dy_m**2 + dz_m**2) < settings.PINCH_THRESHOLD_M:
                pinching_fingers.append(finger)

        if pinching_fingers:
            sum_x = thumb.x + sum(f.x for f in pinching_fingers)
            sum_y = thumb.y + sum(f.y for f in pinching_fingers)
            count = 1 + len(pinching_fingers)
            return True, sum_x / count, sum_y / count

        return False, 0.0, 0.0

    def _get_filtered_landmark(self, landmarks, idx, timestamp, filter_dict):
        """对单个关键点应用 OneEuro 滤波"""
        lm = landmarks[idx]
        if filter_dict is None or timestamp is None:
            return lm
            
        kx, ky, kz = f"lm_{idx}_x", f"lm_{idx}_y", f"lm_{idx}_z"
        kp_config = settings.FILTER_CONFIG['KEYPOINT']
        
        if kx not in filter_dict:
            filter_dict[kx] = OneEuroFilter(kp_config['min_cutoff'], kp_config['beta'], kp_config['d_cutoff'])
            filter_dict[ky] = OneEuroFilter(kp_config['min_cutoff'], kp_config['beta'], kp_config['d_cutoff'])
            filter_dict[kz] = OneEuroFilter(kp_config['min_cutoff'], kp_config['beta'], kp_config['d_cutoff'])
            
        return LandmarkLite(
            filter_dict[kx].filter(lm.x, timestamp),
            filter_dict[ky].filter(lm.y, timestamp),
            filter_dict[kz].filter(lm.z, timestamp)
        )

    def _calculate_orientation(self, p0, p5, p9, p17, hand_label, timestamp, filter_dict):
        """计算手掌的偏航角 (Yaw) 和俯仰角 (Pitch)"""
        v_up = p9 - p0
        v_across = p17 - p5
        
        normal = -np.cross(v_up, v_across)
        if hand_label == "Left":
             normal = -normal
             
        norm_val = np.linalg.norm(normal)
        normal = normal / norm_val if norm_val > 1e-6 else np.array([0.0, 0.0, 1.0])
            
        yaw_deg = math.degrees(math.atan2(normal[0], normal[2]))
        pitch_deg = math.degrees(math.asin(np.clip(-normal[1], -1.0, 1.0)))
        
        if filter_dict is not None and timestamp is not None:
             # 角度滤波
             for angle_type, val in [('yaw', yaw_deg), ('pitch', pitch_deg)]:
                 config = settings.FILTER_CONFIG['HAND'][angle_type.upper()]
                 if angle_type not in filter_dict:
                     filter_dict[angle_type] = OneEuroFilter(config['min_cutoff'], config['beta'], config['d_cutoff'])
                 if angle_type == 'yaw': yaw_deg = filter_dict['yaw'].filter(val, timestamp)
                 else: pitch_deg = filter_dict['pitch'].filter(val, timestamp)
                 
        return yaw_deg, pitch_deg

    def _calculate_grip_factor(self, p0, p9, tips_pts, grip_state):
        """计算握拳程度系数 (0.0=张开, 1.0=握紧)"""
        ref_len = np.linalg.norm(p9 - p0)
        if ref_len < 1e-6: return 0.0
        
        avg_tips_dist = np.mean([np.linalg.norm(pt - p0) for pt in tips_pts])
        ratio = avg_tips_dist / ref_len
        
        # 线性映射 ratio [1.8, 0.9] -> [0.0, 1.0]
        grip_factor = np.clip((1.8 - ratio) / (1.8 - 0.9), 0.0, 1.0)
        
        if grip_state is not None:
            alpha = settings.FILTER_CONFIG['HAND']['DEPTH']['grip_smoothing_alpha']
            grip_factor = alpha * grip_factor + (1.0 - alpha) * grip_state.get('value', 0.0)
            grip_state['value'] = grip_factor
            
        return grip_factor

    def calculate_hand_pos(self, landmarks, frame_width, frame_height, aspect_ratio, timestamp=None, camera_matrix=None, hand_label=None, frame_id=0):
        """
        综合计算手部的 3D 空间位置及旋转状态。
        """
        state = self.hand_filters.get(hand_label, {})
        w_norm_filter = state.get('w_norm')
        pos_filter = state.get('pos')
        filter_dict = state.get('landmarks')
        depth_history = state.get('depth_length_history')
        anchor_state = state.get('depth_anchor')
        grip_state = state.get('grip_state')
        width_corr_state = state.get('width_correction')

        # 1. 关键点获取与滤波
        filtered_lms = {}
        def get_pt(idx):
            if idx not in filtered_lms:
                filtered_lms[idx] = self._get_filtered_landmark(landmarks, idx, timestamp, filter_dict)
            lm = filtered_lms[idx]
            return np.array([lm.x, lm.y * (1.0 / aspect_ratio), lm.z])

        p0, p5, p9, p17 = get_pt(0), get_pt(5), get_pt(9), get_pt(17)
        tips_pts = [get_pt(i) for i in [8, 12, 16, 20]]

        # 2. 姿态计算
        yaw_deg, pitch_deg = self._calculate_orientation(p0, p5, p9, p17, hand_label, timestamp, filter_dict)
        grip_factor = self._calculate_grip_factor(p0, p9, tips_pts, grip_state)

        # 3. 深度估算 (Z)
        focal_length = camera_matrix[0, 0] if camera_matrix is not None else (frame_width / 2.0) / math.tan(math.radians(self.fov) / 2.0)
        
        # 2D 距离计算与滤波
        dist_up_2d = np.linalg.norm(p9[:2] - p0[:2])
        dist_across_2d = np.linalg.norm(p17[:2] - p5[:2])
        
        if filter_dict is not None and timestamp is not None:
            dist_cfg = settings.FILTER_CONFIG['HAND']['PIXEL_DIST']
            for k, val in [('dist_up', dist_up_2d), ('dist_across', dist_across_2d)]:
                if k not in filter_dict:
                    filter_dict[k] = OneEuroFilter(dist_cfg['min_cutoff'], dist_cfg['beta'], dist_cfg['d_cutoff'])
                if k == 'dist_up': dist_up_2d = filter_dict[k].filter(val, timestamp)
                else: dist_across_2d = filter_dict[k].filter(val, timestamp)

        # 几何深度估计
        cos_pitch, cos_yaw = max(0.2, abs(math.cos(math.radians(pitch_deg)))), max(0.2, abs(math.cos(math.radians(yaw_deg))))
        z_up_raw = (focal_length * settings.HAND_REF_LENGTH_M * cos_pitch) / (dist_up_2d * frame_width) if dist_up_2d > 1e-4 else 0.5
        z_across = (focal_length * settings.HAND_REF_WIDTH_M * cos_yaw) / (dist_across_2d * frame_width) if dist_across_2d > 1e-4 else 0.5

        # 宽度修正
        len_corr = width_corr_state.get('value', 1.0) if width_corr_state else 1.0
        depth_cfg = settings.FILTER_CONFIG['HAND']['DEPTH']
        
        if width_corr_state and depth_history and len(depth_history) >= 2:
            motion_score_tmp = min(max(np.std(depth_history) / (np.mean(depth_history) * depth_cfg['sigma_threshold_ratio']), 0.0), 1.0)
            if motion_score_tmp < 0.2 and grip_factor < 0.2 and abs(yaw_deg) < 20.0 and abs(pitch_deg) < 20.0:
                len_corr = 0.05 * np.clip(z_across / z_up_raw, 0.5, 2.0) + 0.95 * len_corr
                width_corr_state['value'] = len_corr
        
        z_up = z_up_raw * len_corr
        w_up, w_across = cos_pitch * (1.0 - grip_factor), cos_yaw
        z_est = (z_up * w_up + z_across * w_across) / (w_up + w_across)
        
        # 4. 深度后处理与锚定
        motion_score = 0.0
        if depth_history is not None:
            depth_history.append(z_up)
            if len(depth_history) > depth_cfg['history_size']: depth_history.pop(0)
            if len(depth_history) >= 2:
                avg_d = np.mean(depth_history)
                motion_score = min(max(np.std(depth_history) / (avg_d * depth_cfg['sigma_threshold_ratio'] if avg_d > 1e-6 else 1e-6), 0.0), 1.0)

        if anchor_state:
            if abs(yaw_deg) < depth_cfg['anchor_yaw_threshold'] and grip_factor < depth_cfg['anchor_grip_threshold']:
                anchor_state.update({'value': z_est, 'frame_id': frame_id, 'timestamp': timestamp or time.time()})
            
            if anchor_state['value'] > 0 and grip_factor > 0:
                w_anchor = min(math.exp(-(frame_id - anchor_state['frame_id']) * 0.693 / depth_cfg['anchor_halflife_frames']) * (1.0 - motion_score) * grip_factor, 0.8)
                z_est = (1.0 - w_anchor) * z_est + w_anchor * anchor_state['value']

        # Z 轴平滑
        if filter_dict is not None and timestamp is not None:
            z_cfg = settings.FILTER_CONFIG['HAND']['Z_VAL']
            if 'z_val' not in filter_dict:
                filter_dict['z_val'] = OneEuroFilter(z_cfg['min_cutoff'], z_cfg['beta'], z_cfg['d_cutoff'])
            z_est = filter_dict['z_val'].filter(z_est, timestamp)

        # 5. 坐标映射 (2D -> 3D)
        lm0, lm9_f = self._get_filtered_landmark(landmarks, 0, timestamp, filter_dict), self._get_filtered_landmark(landmarks, 9, timestamp, filter_dict)
        cx, cy = (camera_matrix[0, 2], camera_matrix[1, 2]) if camera_matrix is not None else (frame_width / 2.0, frame_height / 2.0)
        
        x_est = (((lm0.x + lm9_f.x) / 2.0) * frame_width - cx) * z_est / focal_length
        y_est = (((lm0.y + lm9_f.y) / 2.0) * frame_height - cy) * z_est / focal_length
        
        # 卡尔曼滤波
        if pos_filter:
            r_dynamic = settings.FILTER_CONFIG['HAND']['POSITION']['measurement_noise'] + grip_factor * settings.FILTER_CONFIG['HAND']['POSITION']['r_grip_max'] * (1.0 - motion_score)
            x_est, y_est, z_est = pos_filter.update(x_est, y_est, z_est, R_z=r_dynamic)

        # 6. 捏合检测 (使用滤波后的关键点)
        for tip_idx in [4, 8, 12, 16, 20]:
            get_pt(tip_idx)
        raw_pinch, pinch_cx, pinch_cy = self._detect_pinch_raw(filtered_lms, z_est, aspect_ratio)

        debounce = state.get('pinch_debounce')
        if debounce is not None:
            if raw_pinch == debounce['confirmed']:
                debounce['count'] = 0
            else:
                debounce['count'] += 1
                if debounce['count'] >= settings.PINCH_DEBOUNCE_FRAMES:
                    debounce['confirmed'] = raw_pinch
                    debounce['count'] = 0
            debounce['raw'] = raw_pinch
            is_pinching = debounce['confirmed']
        else:
            is_pinching = raw_pinch

        px, py, pz = 0.0, 0.0, 0.0
        if is_pinching and pinch_cx > 0 and pinch_cy > 0:
            px = (pinch_cx * frame_width - cx) * z_est / focal_length
            py = (pinch_cy * frame_height - cy) * z_est / focal_length
            pz = z_est

        pinch_result = (is_pinching, px, py, pz, pinch_cx, pinch_cy)

        return x_est, y_est, z_est, np.linalg.norm(p17 - p5), yaw_deg, pitch_deg, motion_score, grip_factor, {
            'z_up': z_up, 'z_across': z_across, 'w_up': w_up / (w_up + w_across), 'w_across': w_across / (w_up + w_across), 'len_corr': len_corr
        }, pinch_result

