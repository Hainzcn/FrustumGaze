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

class HandTracker:
    def __init__(self, fov=60.0):
        self.fov = fov
        self.detector = None
        self.roi = None
        self.roi_miss_count = 0
        self.MAX_ROI_MISS_COUNT = 30
        self._init_mediapipe()
        
        # 初始化滤波器状态
        self.hand_filters = {
            'Left': self._create_filter_state(),
            'Right': self._create_filter_state()
        }

    def _create_filter_state(self):
        return {
            'w_norm': OneEuroFilter(
                min_cutoff=settings.HAND_DIST_ONE_EURO_MIN_CUTOFF, 
                beta=settings.HAND_DIST_ONE_EURO_BETA,
                d_cutoff=settings.HAND_DIST_ONE_EURO_D_CUTOFF
            ),
            'pos': Simple3DKalmanFilter(
                process_noise=settings.HAND_KALMAN_PROCESS_NOISE, 
                measurement_noise=settings.HAND_KALMAN_MEASUREMENT_NOISE
            ),
            'depth_length_history': [],
            'depth_anchor': {'value': 0.0, 'frame_id': 0, 'timestamp': 0.0},
            'width_correction': {'value': 1.0, 'count': 0},
            'grip_state': {'value': 0.0},
            'landmarks': {} # OneEuroFilter dict
        }

    def _init_mediapipe(self):
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
            print(f"HandTracker: MediaPipe Initialized. FOV={self.fov}")
        except Exception as e:
            print(f"HandTracker: Failed to init MediaPipe: {e}")
            raise e

    def process(self, mp_image, timestamp_ms):
        if not self.detector:
            return None
        return self.detector.detect_for_video(mp_image, timestamp_ms)

    def close(self):
        if self.detector:
            self.detector.close()

    def get_filter_state(self, label):
        if label in self.hand_filters:
            return self.hand_filters[label]
        return None

    def calculate_roi(self, landmarks_list, padding_factor=0.5):
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
        overlap_1 = inter_area / area1 if area1 > 0 else 0
        overlap_2 = inter_area / area2 if area2 > 0 else 0
        max_overlap = max(overlap_1, overlap_2)

        return iou, max_overlap

    def filter_overlapping_hands(self, landmarks_list, handedness_list, iou_threshold=0.5, overlap_threshold=0.7):
        """
        过滤重叠严重的手部检测结果
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
                iou, max_overlap = self._calculate_iou(bboxes[i], bboxes[j])
                if iou > iou_threshold or max_overlap > overlap_threshold:
                    is_kept = False 
                    break
            
            if is_kept:
                final_indices.append(i)
        
        filtered_landmarks = [landmarks_list[i] for i in final_indices]
        filtered_handedness = [handedness_list[i] for i in final_indices]
        
        return filtered_landmarks, filtered_handedness

    def detect_pinch(self, landmarks, z_depth, aspect_ratio):
        """
        检测是否捏起 (拇指与其他手指)
        返回: (is_pinching, pinch_x, pinch_y, pinch_z, pinch_cx, pinch_cy)
        """
        THUMB_TIP = 4
        INDEX_TIP = 8
        MIDDLE_TIP = 12
        RING_TIP = 16
        PINKY_TIP = 20
        
        TIPS = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
        
        thumb = landmarks[THUMB_TIP]
        
        PINCH_THRESHOLD_M = settings.PINCH_THRESHOLD_M
        tan_half_fov = math.tan(math.radians(self.fov) / 2.0)
        
        pinching_fingers = []
        
        for tip_idx in TIPS:
            finger = landmarks[tip_idx]
            dx = thumb.x - finger.x
            dy = thumb.y - finger.y
            
            dx_m = dx * z_depth * 2.0 * tan_half_fov
            dy_m = dy * z_depth * (1.0 / aspect_ratio) * 2.0 * tan_half_fov
            
            dist_m = math.sqrt(dx_m*dx_m + dy_m*dy_m)
            
            if dist_m < PINCH_THRESHOLD_M:
                pinching_fingers.append(finger)
        
        if len(pinching_fingers) > 0:
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

    def calculate_hand_pos(self, landmarks, frame_width, frame_height, aspect_ratio, timestamp=None, camera_matrix=None, hand_label=None, frame_id=0):
        """
        计算手部空间位置 (Camera Space) 和 Yaw 角
        """
        w_norm_filter = None
        pos_filter = None
        one_euro_filter_dict = None
        depth_length_history = None
        anchor_state = None
        grip_state = None
        width_correction_state = None

        if hand_label in self.hand_filters:
             state = self.hand_filters[hand_label]
             w_norm_filter = state['w_norm']
             pos_filter = state['pos']
             depth_length_history = state['depth_length_history']
             anchor_state = state['depth_anchor']
             grip_state = state['grip_state']
             width_correction_state = state['width_correction']
             one_euro_filter_dict = state['landmarks']
        
        def get_pt_unified(lm):
            return np.array([lm.x, lm.y * (1.0 / aspect_ratio), lm.z])

        p0 = get_pt_unified(landmarks[0])   # Wrist
        p5 = get_pt_unified(landmarks[5])   # Index MCP
        p9 = get_pt_unified(landmarks[9])   # Middle MCP
        p17 = get_pt_unified(landmarks[17]) # Pinky MCP
        
        v_up = p9 - p0
        v_across = p17 - p5
        
        normal = np.cross(v_up, v_across)
        normal = -normal
        
        if hand_label == "Left":
             normal = -normal
             
        norm_val = np.linalg.norm(normal)
        if norm_val > 1e-6:
            normal /= norm_val
        else:
            normal = np.array([0.0, 0.0, 1.0])
            
        yaw = math.atan2(normal[0], normal[2])
        yaw_deg = math.degrees(yaw)
        
        # Grip Factor
        ref_len_grip = np.linalg.norm(p9 - p0)
        grip_factor = 0.0
        
        if ref_len_grip > 1e-6:
            tips_indices = [8, 12, 16, 20]
            tips_dist_sum = 0.0
            for idx in tips_indices:
                pt = get_pt_unified(landmarks[idx])
                tips_dist_sum += np.linalg.norm(pt - p0)
            
            avg_tips_dist = tips_dist_sum / 4.0
            ratio = avg_tips_dist / ref_len_grip
            
            r_open = 1.8
            r_closed = 0.9
            
            grip_factor = (r_open - ratio) / (r_open - r_closed)
            grip_factor = max(0.0, min(grip_factor, 1.0))
            
            if grip_state is not None:
                alpha = settings.HAND_GRIP_SMOOTHING_ALPHA
                last_grip = grip_state.get('value', 0.0)
                grip_factor = alpha * grip_factor + (1.0 - alpha) * last_grip
                grip_state['value'] = grip_factor

        # 3D Position Estimation
        if camera_matrix is None:
             focal_length = (frame_width / 2.0) / math.tan(math.radians(self.fov) / 2.0)
        else:
             focal_length = camera_matrix[0, 0]

        REF_LENGTH_UP_M = settings.HAND_REF_LENGTH_M
        REF_LENGTH_ACROSS_M = settings.HAND_REF_WIDTH_M
        
        pitch = math.asin(np.clip(-normal[1], -1.0, 1.0))
        pitch_deg = math.degrees(pitch)
        
        dist_up_2d = np.linalg.norm(p9[:2] - p0[:2])
        
        if dist_up_2d < 1e-4:
            z_up_raw = 0.5
        else:
            cos_pitch = max(0.2, abs(math.cos(pitch))) 
            len_px_up = dist_up_2d * frame_width
            z_up_raw = (focal_length * REF_LENGTH_UP_M * cos_pitch) / len_px_up

        dist_across_2d = np.linalg.norm(p17[:2] - p5[:2])
        
        if dist_across_2d < 1e-4:
            z_across = 0.5
        else:
            cos_yaw = max(0.2, abs(math.cos(yaw)))
            len_px_across = dist_across_2d * frame_width
            z_across = (focal_length * REF_LENGTH_ACROSS_M * cos_yaw) / len_px_across
            
        length_correction_factor = 1.0
        if width_correction_state is not None:
            length_correction_factor = width_correction_state.get('value', 1.0)
            
            temp_motion_score = 1.0
            if depth_length_history and len(depth_length_history) >= 2:
                 sigma = np.std(depth_length_history)
                 avg = np.mean(depth_length_history)
                 if avg > 1e-6:
                     temp_motion_score = min(max(sigma / (avg * settings.HAND_DEPTH_SIGMA_THRESHOLD_RATIO), 0.0), 1.0)
            
            can_update_correction = (temp_motion_score < 0.2) and \
                                    (grip_factor < 0.2) and \
                                    (abs(yaw_deg) < 20.0) and \
                                    (abs(pitch_deg) < 20.0)
            
            if can_update_correction:
                target_factor = z_across / z_up_raw
                target_factor = max(0.5, min(target_factor, 2.0))
                
                alpha_corr = 0.05
                length_correction_factor = alpha_corr * target_factor + (1.0 - alpha_corr) * length_correction_factor
                width_correction_state['value'] = length_correction_factor
        
        z_up = z_up_raw * length_correction_factor

        w_up = max(0.2, abs(math.cos(pitch)))
        w_across = max(0.2, abs(math.cos(yaw)))
        
        w_up *= (1.0 - grip_factor)

        w_sum = w_up + w_across
        z_est = (z_up * w_up + z_across * w_across) / w_sum
        
        depth_details = {
            'z_up': z_up,
            'z_across': z_across,
            'w_up': w_up / w_sum,
            'w_across': w_across / w_sum,
            'len_corr': length_correction_factor
        }
        
        center_x_norm = (landmarks[0].x + landmarks[9].x) / 2.0
        center_y_norm = (landmarks[0].y + landmarks[9].y) / 2.0
        
        center_x_px = center_x_norm * frame_width
        center_y_px = center_y_norm * frame_height
        
        cx = frame_width / 2.0
        cy = frame_height / 2.0
        if camera_matrix is not None:
            cx = camera_matrix[0, 2]
            cy = camera_matrix[1, 2]
            
        x_est = (center_x_px - cx) * z_est / focal_length
        y_est = (center_y_px - cy) * z_est / focal_length
        
        motion_score = 0.0
        if depth_length_history is not None:
            depth_length_history.append(z_up)
            if len(depth_length_history) > settings.HAND_DEPTH_HISTORY_SIZE:
                depth_length_history.pop(0)
            
            if len(depth_length_history) >= 2:
                sigma_length = np.std(depth_length_history)
                avg_depth = np.mean(depth_length_history)
                sigma_threshold = avg_depth * settings.HAND_DEPTH_SIGMA_THRESHOLD_RATIO
                if sigma_threshold < 1e-6: sigma_threshold = 1e-6
                motion_score = min(max(sigma_length / sigma_threshold, 0.0), 1.0)

        r_base = settings.HAND_KALMAN_R_BASE
        r_grip_max = settings.HAND_KALMAN_R_GRIP_MAX
        r_dynamic = r_base + grip_factor * r_grip_max * (1.0 - motion_score)

        anchor_weight = 0.0
        anchor_depth = 0.0
        
        if anchor_state is not None:
            can_update_anchor = (abs(yaw_deg) < settings.HAND_DEPTH_ANCHOR_YAW_THRESHOLD) and \
                                (grip_factor < settings.HAND_DEPTH_ANCHOR_GRIP_THRESHOLD)
                                
            if can_update_anchor:
                anchor_state['value'] = z_est
                anchor_state['frame_id'] = frame_id
                anchor_state['timestamp'] = timestamp if timestamp else time.time()
                
            if anchor_state['value'] > 0 and grip_factor > 0:
                frames_since_update = frame_id - anchor_state['frame_id']
                if frames_since_update < 0: frames_since_update = 0
                
                half_life = settings.HAND_DEPTH_ANCHOR_HALFLIFE_FRAMES
                decay = math.exp(-frames_since_update * 0.693 / half_life)
                
                motion_factor = 1.0 - motion_score
                w_anchor = decay * motion_factor * grip_factor
                w_anchor = min(w_anchor, 0.8) 
                
                anchor_weight = w_anchor
                anchor_depth = anchor_state['value']
                
                z_est = (1.0 - w_anchor) * z_est + w_anchor * anchor_depth

        if one_euro_filter_dict is not None and timestamp is not None:
            if 'yaw' not in one_euro_filter_dict:
                one_euro_filter_dict['yaw'] = OneEuroFilter(
                    min_cutoff=settings.HAND_YAW_ONE_EURO_MIN_CUTOFF, 
                    beta=settings.HAND_YAW_ONE_EURO_BETA,
                    d_cutoff=settings.HAND_YAW_ONE_EURO_D_CUTOFF
                )
            yaw_deg = one_euro_filter_dict['yaw'].filter(yaw_deg, timestamp)
            
        if pos_filter:
            x_est, y_est, z_est = pos_filter.update(x_est, y_est, z_est, R_z=r_dynamic)
            
        w_norm = np.linalg.norm(p17 - p5)
        
        return x_est, y_est, z_est, w_norm, yaw_deg, pitch_deg, motion_score, grip_factor, depth_details
