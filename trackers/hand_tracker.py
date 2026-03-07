
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
from utils.math_utils import OneEuroFilter, Simple3DKalmanFilter, AdaptiveEKF
from utils.image_utils import GlobalImagePreprocessor
from config import settings

# 定义简单的 Landmark 类以便于 Pickle
class LandmarkLite:
    def __init__(self, x, y, z, visibility=0.0, presence=0.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility
        self.presence = presence

# 定义简单的 Result 类以便于 Pickle
class HandDetectionResultLite:
    def __init__(self, hand_landmarks_list, handedness_list=None):
        self.multi_hand_landmarks = []
        self.multi_handedness = []
        
        if hand_landmarks_list:
            for landmarks in hand_landmarks_list:
                simple_landmarks = []
                for lm in landmarks:
                    # Check if lm has visibility/presence (MediaPipe landmarks do)
                    vis = getattr(lm, 'visibility', 0.0)
                    if vis is None: vis = 0.0
                    pres = getattr(lm, 'presence', 0.0)
                    if pres is None: pres = 0.0
                    simple_landmarks.append(LandmarkLite(lm.x, lm.y, lm.z, vis, pres))
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
        
        # 几何一致性历史记录 {label: [area1, area2, ...]}
        self.hand_area_history = {}
        self.MAX_HISTORY_LEN = 30

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

    def _calculate_landmark_confidence(self, landmarks):
        """
        计算关键点置信度均值 (MediaPipe visibility)
        取手掌区域关键点 (0, 5, 9, 13, 17)
        """
        indices = [0, 5, 9, 13, 17]
        total_vis = 0.0
        count = 0
        for i in indices:
            if i < len(landmarks):
                # LandmarkLite or MediaPipe Landmark
                # getattr might return None if attribute exists but is None
                vis = getattr(landmarks[i], 'visibility', 0.0)
                if vis is None:
                    vis = 0.0
                total_vis += vis
                count += 1
        
        if count == 0:
            return 0.0
        return total_vis / count

    def _calculate_polygon_area(self, points):
        """Shoelace formula for polygon area"""
        area = 0.0
        n = len(points)
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]
        return abs(area) / 2.0

    def _calculate_geometric_consistency(self, landmarks, label, aspect_ratio):
        """
        几何一致性校验 (改进版)
        计算当前帧手掌形状因子 (Area / (Width^2)) 与前 N 帧均值的偏差比
        消除距离缩放带来的影响
        返回: score, shape_factor
        """
        indices = [0, 5, 9, 13, 17]
        points = []
        for i in indices:
            if i < len(landmarks):
                # Apply aspect ratio to y to make shape calculation uniform
                points.append((landmarks[i].x, landmarks[i].y / aspect_ratio))
        
        if len(points) < 5:
            return 1.0, 0.0 # Not enough points
            
        current_area = self._calculate_polygon_area(points)
        
        # Calculate characteristic length (Wrist to Middle MCP)
        p0 = points[0]
        p9 = points[2] # Index 2 in points corresponds to landmark 9
        dx = p0[0] - p9[0]
        dy = p0[1] - p9[1]
        length_sq = dx*dx + dy*dy
        
        if length_sq < 1e-9:
            return 0.0, 0.0 # Degenerate hand
            
        shape_factor = current_area / length_sq
        
        # 如果没有 label 或者 label 为 Unknown，暂不记录历史或者只使用临时历史
        if not label or label == "Unknown":
            return 1.0, shape_factor
            
        if label not in self.hand_area_history:
            self.hand_area_history[label] = []
            
        history = self.hand_area_history[label]
        
        if not history:
            history.append(shape_factor)
            return 1.0, shape_factor
            
        mean_factor = sum(history) / len(history)
        
        # Update history
        history.append(shape_factor)
        if len(history) > self.MAX_HISTORY_LEN:
            history.pop(0)
            
        if mean_factor < 1e-9:
            return 0.0, shape_factor
            
        deviation = abs(shape_factor - mean_factor) / mean_factor
        
        # 偏差超过 30% 判定为几何畸变
        # 映射到 [0, 1] 分数
        score = max(0.0, 1.0 - (deviation / 0.3))
        return score, shape_factor

    def _calculate_reprojection_error(self, image_points, object_points, rvec, tvec, camera_matrix, dist_coeffs):
        """
        计算重投影误差 RMSE (像素)
        """
        try:
            projected_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
            # image_points shape: (N, 2), projected_points shape: (N, 1, 2)
            projected_points = projected_points.reshape(-1, 2)
            
            error = cv2.norm(image_points, projected_points, cv2.NORM_L2)
            rmse = error / math.sqrt(len(image_points))
            return rmse
        except Exception:
            return 100.0 # High error on failure

    def _calculate_hand_openness(self, landmarks, aspect_ratio):
        """
        计算手掌张开程度 (Openness)
        通过比较 (Wrist->MiddleTip) 和 (Wrist->MiddleMCP) 的距离
        用于过滤握拳等非展开手势
        """
        # 0: Wrist, 9: Middle MCP, 12: Middle Tip
        if len(landmarks) <= 12:
            return 0.0, 0.0
            
        p0 = landmarks[0]
        p9 = landmarks[9]
        p12 = landmarks[12]
        
        # Calculate 3D distances (assuming z is same scale as x)
        # Apply aspect ratio to y for correct Euclidean distance in 3D volume approximation
        
        def dist3d(p_a, p_b):
            dx = p_a.x - p_b.x
            dy = (p_a.y - p_b.y) / aspect_ratio
            dz = p_a.z - p_b.z
            return math.sqrt(dx*dx + dy*dy + dz*dz)

        d_wrist_mcp = dist3d(p0, p9)
        d_wrist_tip = dist3d(p0, p12)
        
        if d_wrist_mcp < 1e-6:
            return 0.0, 0.0
            
        ratio = d_wrist_tip / d_wrist_mcp
        
        # Thresholds:
        # Open hand: Ratio ~ 2.0 or higher (fingers extended)
        # Fist: Ratio ~ 1.0 or lower (fingers curled)
        # Relaxed: ~ 1.5
        
        # Map 1.2 -> 0.0, 1.6 -> 1.0 (Strictly penalize fists)
        score = np.clip((ratio - 1.2) / (1.6 - 1.2), 0.0, 1.0)
        return score, ratio

    def _estimate_normal_svd(self, landmarks, frame_width, frame_height):
        """
        通道 B: SVD 平面拟合估算法向量
        """
        # Indices: Wrist, IndexMCP, MiddleMCP, RingMCP, PinkyMCP
        indices = [0, 5, 9, 13, 17]
        points = []
        for i in indices:
            if i < len(landmarks):
                lm = landmarks[i]
                # Convert to consistent 3D space (Image Pixel Space)
                # Z is relative to wrist, scale is approx same as X
                # We use width for Z scale to match X
                points.append([lm.x * frame_width, lm.y * frame_height, lm.z * frame_width])
        
        if len(points) < 3:
            return np.array([0.0, 0.0, -1.0])

        points = np.array(points)
        # Center the points
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        
        # SVD
        try:
            u, s, vh = np.linalg.svd(centered)
            # Normal is the last row of vh (corresponding to smallest singular value)
            normal = vh[-1]
            
            # Normalize
            norm_val = np.linalg.norm(normal)
            if norm_val > 1e-6:
                normal /= norm_val
            else:
                return np.array([0.0, 0.0, -1.0])
                
            return normal
        except Exception:
            return np.array([0.0, 0.0, -1.0])

    def _calculate_hand_pos(self, landmarks, frame_width, frame_height, aspect_ratio, w_norm_filter=None, pos_filter=None, normal_filter=None, one_euro_filter_dict=None, timestamp=None, camera_matrix=None, label="Unknown", score=0.0):
        """
        计算手部空间位置 (Camera Space) 并评估置信度 q
        多源融合法向量估计 (PnP + SVD + Prediction)
        """
        # --- 1. Landmark Confidence Check ---
        conf_handedness = np.clip((score - 0.4) / (0.8 - 0.4), 0.0, 1.0)
        conf_openness, openness_ratio = self._calculate_hand_openness(landmarks, aspect_ratio)
        conf_lm_score = conf_handedness * conf_openness

        # --- 2. Geometric Consistency Check ---
        conf_geo_score, shape_factor = self._calculate_geometric_consistency(landmarks, label, aspect_ratio)

        # 3D Model Points (Meters)
        scale = settings.HAND_PALM_WIDTH_CM / 6.0
        
        # 0: Wrist, 5: Index, 9: Middle, 13: Ring, 17: Pinky
        # Model defined in local frame with Z=0 (Flat Palm)
        # Normal of model points is (0, 0, -1) [Right-handed rule with current point order?]
        # Let's verify model normal direction later or ensure consistency.
        model_points = np.array([
            (0.08 * scale, 0.03 * scale, 0.0),     # 0: Wrist
            (0.0, -0.03 * scale, 0.0),             # 5: Index MCP
            (-0.01 * scale, -0.01 * scale, 0.0),   # 9: Middle MCP
            (-0.005 * scale,  0.01 * scale, 0.0),  # 13: Ring MCP
            (0.0,  0.03 * scale, 0.0)              # 17: Pinky MCP
        ], dtype="double")
        
        # OneEuroFilter for landmarks
        p_coords = []
        indices = [0, 5, 9, 13, 17]
        for idx in indices:
            px, py = landmarks[idx].x, landmarks[idx].y
            if one_euro_filter_dict is not None and timestamp is not None:
                def get_filtered_val(name, val):
                    if name not in one_euro_filter_dict:
                        one_euro_filter_dict[name] = OneEuroFilter(
                            min_cutoff=settings.HAND_POS_ONE_EURO_MIN_CUTOFF, 
                            beta=settings.HAND_POS_ONE_EURO_BETA,
                            d_cutoff=settings.HAND_POS_ONE_EURO_D_CUTOFF
                        )
                    return one_euro_filter_dict[name].filter(val, timestamp)
                px = get_filtered_val(f'p{idx}_x', px)
                py = get_filtered_val(f'p{idx}_y', py)
            p_coords.append((px * frame_width, py * frame_height))

        image_points = np.array(p_coords, dtype="double")
        
        if camera_matrix is None:
            focal_length = (frame_width / 2.0) / math.tan(math.radians(self.fov) / 2.0)
            center = (frame_width / 2.0, frame_height / 2.0)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype="double"
            )
        dist_coeffs = np.zeros((4, 1))

        # --- Channel A: PnP Solver ---
        # Try IPPE first (4 points: 0, 5, 13, 17)
        model_points_4 = model_points[[0, 1, 3, 4]]
        image_points_4 = image_points[[0, 1, 3, 4]]
        
        success = False
        rvecs, tvecs = [], []
        
        try:
            pnp_result = cv2.solvePnPGeneric(
                model_points_4, image_points_4, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE
            )
            if len(pnp_result) >= 3:
                rvecs, tvecs = pnp_result[1], pnp_result[2]
                success = len(rvecs) > 0
        except Exception:
            success = False

        if not success:
            try:
                success_iter, rvec_iter, tvec_iter = cv2.solvePnP(
                    model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
                )
                if success_iter:
                    rvecs, tvecs = [rvec_iter], [tvec_iter]
                    success = True
            except Exception:
                success = False

        best_rvec, best_tvec = None, None
        
        # Calculate Model Normal (used for disambiguation and PnP normal extraction)
        v1_model = model_points[1] - model_points[0]
        v2_model = model_points[4] - model_points[0]
        normal_model = np.cross(v1_model, v2_model) # Roughly (0, 0, -1)
        if np.linalg.norm(normal_model) > 1e-6:
            normal_model /= np.linalg.norm(normal_model)

        if success:
            # Disambiguation using observed normal from raw landmarks
            # Calculate observed normal in Camera Space
            def get_vec(lm_to, lm_from):
                return np.array([
                    lm_to.x - lm_from.x,
                    (lm_to.y - lm_from.y) * (frame_height / frame_width), 
                    lm_to.z - lm_from.z
                ])
            v1_obs = get_vec(landmarks[5], landmarks[0])
            v2_obs = get_vec(landmarks[17], landmarks[0])
            normal_obs = np.cross(v1_obs, v2_obs)
            if np.linalg.norm(normal_obs) > 1e-6:
                normal_obs /= np.linalg.norm(normal_obs)

            max_dot = -100.0
            for i in range(len(rvecs)):
                rvec = rvecs[i]
                tvec = tvecs[i]
                R, _ = cv2.Rodrigues(rvec)
                normal_cam = np.dot(R, normal_model)
                dot_prod = np.dot(normal_cam, normal_obs)
                if dot_prod > max_dot:
                    max_dot = dot_prod
                    best_rvec = rvec
                    best_tvec = tvec

        # --- 3. Reprojection Error Check ---
        if best_rvec is not None:
            reproj_rmse = self._calculate_reprojection_error(image_points, model_points, best_rvec, best_tvec, camera_matrix, dist_coeffs)
        else:
            reproj_rmse = 100.0

        if reproj_rmse < 10.0:
            conf_rep_score = 1.0
        elif reproj_rmse > 30.0:
            conf_rep_score = 0.0
        else:
            conf_rep_score = 1.0 - (reproj_rmse - 10.0) / (30.0 - 10.0)

        # --- Final Q Calculation ---
        q = conf_lm_score * conf_geo_score * conf_rep_score

        # --- Multi-Source Normal Fusion ---
        
        # 1. Channel A: PnP Normal
        n_A = np.array([0.0, 0.0, 0.0])
        if best_rvec is not None:
            R, _ = cv2.Rodrigues(best_rvec)
            n_A = np.dot(R, normal_model)
        
        # 2. Channel B: SVD Plane Normal
        n_B = self._estimate_normal_svd(landmarks, frame_width, frame_height)
        # Ensure n_B aligns with n_A (or generally towards camera if n_A missing)
        # If n_A is valid, align n_B to n_A
        if np.linalg.norm(n_A) > 0.1:
            if np.dot(n_A, n_B) < 0:
                n_B = -n_B
        else:
            # Fallback: Assume palm faces camera (Z < 0 in OpenCV frame)
            # But wait, normal_model is approx (0,0,-1).
            # If hand is facing camera, normal should be approx (0,0,-1).
            if n_B[2] > 0: # If pointing away from camera
                n_B = -n_B
                
        # 3. Channel C: Prediction
        n_C = np.array([0.0, 0.0, -1.0]) # Default fallback
        if normal_filter:
            if not normal_filter.first_run:
                # Use prediction from previous step (AdaptiveEKF)
                pred_theta, pred_phi = normal_filter.predict()
                nz_p = math.cos(pred_theta)
                nx_p = math.sin(pred_theta) * math.cos(pred_phi)
                ny_p = math.sin(pred_theta) * math.sin(pred_phi)
                n_C = np.array([nx_p, ny_p, nz_p])
        
        # Weights
        # wA = q * alphaA
        # wB = betaB
        # wC = (1-q) * alphaC
        alpha_A = 1.0
        beta_B = 0.5
        alpha_C = 0.8
        
        wA = q * alpha_A
        wB = beta_B
        wC = (1.0 - q) * alpha_C
        
        # Fusion
        n_final_raw = wA * n_A + wB * n_B + wC * n_C
        norm_final = np.linalg.norm(n_final_raw)
        
        if norm_final > 1e-6:
            n_final = n_final_raw / norm_final
        else:
            n_final = n_C # Fallback
            
        # Update Filter with the Fused Result (Feedback Loop)
        if normal_filter:
            # Convert n_final (Observation) to Spherical (theta, phi)
            nx, ny, nz = n_final[0], n_final[1], n_final[2]
            
            # Theta: [0, pi], from Z axis
            theta_obs = math.acos(np.clip(nz, -1.0, 1.0))
            phi_obs = math.atan2(ny, nx)
            
            # Update AdaptiveEKF (Correct Step)
            # R is dynamically adjusted by q
            
            # Check for NaN/Inf in observations
            if math.isnan(theta_obs) or math.isnan(phi_obs):
                # Skip update if observation is invalid
                pass
            else:
                theta_f, phi_f = normal_filter.correct(theta_obs, phi_obs, q)
                
                # Check for NaN in output
                if not (math.isnan(theta_f) or math.isnan(phi_f)):
                    # Use Filtered Result as Final Output
                    nz_f = math.cos(theta_f)
                    nx_f = math.sin(theta_f) * math.cos(phi_f)
                    ny_f = math.sin(theta_f) * math.sin(phi_f)
                    n_final = np.array([nx_f, ny_f, nz_f])
        
        # --- Calculate Output Variables ---
        
        # Position (from PnP or keep previous?)
        if best_tvec is not None:
            x = best_tvec[0][0]
            y = best_tvec[1][0]
            z = best_tvec[2][0]
        else:
            # If PnP failed, we don't have a new position.
            # Return 0 or previous? 
            # Existing logic returned 0.0.
            x, y, z = 0.0, 0.0, 0.0
            
        # Yaw from n_final
        # n_final is in Camera Space.
        # Yaw is rotation around Y axis.
        # Project n_final to XZ plane.
        # Normal (0, 0, -1) -> Yaw 0.
        # Normal (-1, 0, 0) -> Yaw +90 (Points Left)
        # Normal (1, 0, 0) -> Yaw -90 (Points Right)
        # yaw = atan2(-nx, -nz)
        yaw = math.atan2(-n_final[0], -n_final[2])
        yaw_deg = math.degrees(yaw)
        
        # Width (w_norm)
        dx = landmarks[5].x - landmarks[17].x
        dy = (landmarks[5].y - landmarks[17].y) * (1.0 / aspect_ratio)
        w_norm = math.sqrt(dx*dx + dy*dy)
        
        # Pos Filter
        if pos_filter and best_tvec is not None:
            x, y, z = pos_filter.update(x, y, z)
            
        scores_tuple = (conf_lm_score, conf_geo_score, conf_rep_score, score, shape_factor, reproj_rmse, openness_ratio)
        
        return x, y, z, w_norm, yaw_deg, q, scores_tuple


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
                ),
                'normal': AdaptiveEKF(process_noise=1e-5, measurement_noise_base=1e-3)
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
                ),
                'normal': AdaptiveEKF(process_noise=1e-5, measurement_noise_base=1e-3)
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
                
                # --- 手部检测 ---
                
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
                                
                                vis = getattr(lm, 'visibility', 0.0)
                                if vis is None: vis = 0.0
                                pres = getattr(lm, 'presence', 0.0)
                                if pres is None: pres = 0.0
                                mapped_landmarks.append(LandmarkLite(norm_x, norm_y, lm.z, vis, pres))
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
                        normal_filter = None
                        one_euro_filter_dict = None
                        label = "Unknown"
                        score = 0.0 # Default score
                        
                        if result_lite.multi_handedness and idx < len(result_lite.multi_handedness):
                            # handedness[0] is the category with highest score
                            categories = result_lite.multi_handedness[idx]
                            if categories:
                                label = categories[0]['label'] # "Left" or "Right"
                                score = categories[0]['score'] # Get score
                                if label in self.hand_filters:
                                    w_norm_filter = self.hand_filters[label]['w_norm']
                                    pos_filter = self.hand_filters[label]['pos']
                                    normal_filter = self.hand_filters[label]['normal']
                                    # 检查是否有 OneEuroFilter 字典用于关键点滤波
                                    if 'landmarks' not in self.hand_filters[label]:
                                        self.hand_filters[label]['landmarks'] = {}
                                    one_euro_filter_dict = self.hand_filters[label]['landmarks']
                        
                        x, y, z, w_norm, yaw, q, scores = self._calculate_hand_pos(
                            landmarks, target_w, target_h, aspect_ratio, 
                            w_norm_filter=w_norm_filter, 
                            pos_filter=pos_filter, 
                            normal_filter=normal_filter,
                            one_euro_filter_dict=one_euro_filter_dict,
                            timestamp=timestamp_ms / 1000.0,
                            camera_matrix=cached_camera_matrix,
                            label=label,
                            score=score
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
                                'q': q,
                                'scores': scores, # (lm, geo, rep)
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
                import traceback
                print(f"Processing Error in Hand Process: {e}")
                traceback.print_exc()

        # 清理
        detector.close()
        for mgr in self.shm_managers:
            try:
                mgr.close()
            except:
                pass
