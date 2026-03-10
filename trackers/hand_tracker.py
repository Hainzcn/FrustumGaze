
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

    def _calculate_hand_pos(self, landmarks, frame_width, frame_height, aspect_ratio, w_norm_filter=None, pos_filter=None, one_euro_filter_dict=None, timestamp=None, camera_matrix=None, hand_label=None, depth_length_history=None, anchor_state=None, grip_state=None, width_correction_state=None, frame_id=0):
        """
        计算手部空间位置 (Camera Space) 和 Yaw 角
        完全弃用 PnP，改用几何特征计算
        """
        
        # 1. 几何法计算 Yaw (Rotation around Y-axis)
        # 使用 Wrist(0), Index MCP(5), Middle MCP(9), Pinky MCP(17)
        # 构建统一坐标系 (以 Width 为基准单位)
        
        def get_pt_unified(lm):
            # 将 y 轴缩放以匹配 x 轴的比例 (假设 square pixels)
            # lm.x 是 0..1 (relative to width)
            # lm.y 是 0..1 (relative to height)
            # unified_y = lm.y * (height / width) = lm.y * (1.0 / aspect_ratio)
            # lm.z 已经是 relative to width (MediaPipe spec)
            return np.array([lm.x, lm.y * (1.0 / aspect_ratio), lm.z])

        p0 = get_pt_unified(landmarks[0])   # Wrist
        p5 = get_pt_unified(landmarks[5])   # Index MCP
        p9 = get_pt_unified(landmarks[9])   # Middle MCP
        p17 = get_pt_unified(landmarks[17]) # Pinky MCP
        
        # 向量 1: 手腕 -> 中指根部 (大致指向手指方向 / Up direction)
        v_up = p9 - p0
        
        # 向量 2: 食指根部 -> 小指根部 (手掌横向 / Across direction)
        v_across = p17 - p5
        
        # 计算法向量 (Normal)
        # 根据用户反馈，当手心朝向摄像头时，观测到的 Cross Product 结果是指向 -Z (Out of screen)
        # 导致 Yaw ~ 180 度 (-170)。
        # 用户期望：手心朝向摄像头时 Yaw = 0 (Normal 指向 +Z, Into screen)
        # 因此我们需要反转 Cross Product 的结果
        
        # Cross Product
        normal = np.cross(v_up, v_across)
        
        # 左右手坐标系处理
        # 1. 基础修正: 将法向量反转，使其从指向相机变为背离相机 (Into Screen)
        #    这样 Right Hand Palm to Camera 会产生 0 度 (Instead of 180)
        #    Right Hand Face Right 会产生 +90 (Instead of -90)
        #    Right Hand Face Left 会产生 -90 (Instead of +90)
        normal = -normal
        
        # 2. 左右手对称性修正
        #    Left Hand Palm to Camera:
        #    Thumb(Right/LargeX), Pinky(Left/SmallX) -> v_across (+X)
        #    v_up (-Y)
        #    raw_normal = (-Y) x (+X) = +Z (0 deg)
        #    inverted_normal = -Z (180 deg)
        #    所以左手需要再次反转，或者保持原始 raw_normal (即不进行第一步反转)
        #    为了代码清晰，我们显式处理：
        
        if hand_label == "Left":
             # 左手不需要上述的全局反转 (因为它本身就是 +Z)
             # 或者说，我们需要再一次反转回来
             normal = -normal
             
        # 总结逻辑简化:
        # Right Hand: normal = -cross(v_up, v_across)
        # Left Hand:  normal =  cross(v_up, v_across)
            
        # 归一化
        norm_val = np.linalg.norm(normal)
        if norm_val > 1e-6:
            normal /= norm_val
        else:
            normal = np.array([0.0, 0.0, 1.0])
            
        # 计算 Yaw
        # Project onto X-Z plane.
        # normal = (nx, ny, nz)
        # Yaw = atan2(nx, nz)
        # Check: Right Hand Facing Camera -> N=(0,0,1) -> atan2(0, 1) = 0 deg.
        # Rotated Right (Thumb to Camera, Palm Left) -> N=(-1,0,0) -> atan2(-1, 0) = -90.
        # Rotated Left (Pinky to Camera, Palm Right) -> N=(1,0,0) -> atan2(1, 0) = 90.
        yaw = math.atan2(normal[0], normal[2])
        yaw_deg = math.degrees(yaw)
        
        # 1.5 聚拢系数 (Grip Factor) 计算
        # 提前计算以便用于深度融合权重调整
        # 使用 Wrist(0) 到 Middle MCP(9) 作为参考长度
        ref_len_grip = np.linalg.norm(p9 - p0)
        grip_factor = 0.0
        
        if ref_len_grip > 1e-6:
            tips_indices = [8, 12, 16, 20] # Index, Middle, Ring, Pinky tips
            tips_dist_sum = 0.0
            for idx in tips_indices:
                pt = get_pt_unified(landmarks[idx])
                tips_dist_sum += np.linalg.norm(pt - p0)
            
            avg_tips_dist = tips_dist_sum / 4.0
            ratio = avg_tips_dist / ref_len_grip
            
            # 经验阈值: 展开时 ratio ~ 1.8+, 握拳时 ratio ~ 0.9
            r_open = 1.8
            r_closed = 0.9
            
            # 归一化到 [0, 1], 0=Open, 1=Closed
            grip_factor = (r_open - ratio) / (r_open - r_closed)
            grip_factor = max(0.0, min(grip_factor, 1.0))
            
            # EMA 平滑
            if grip_state is not None:
                # 获取上一帧的 grip_factor
                last_grip = grip_state.get('value', 0.0)
                # 应用 EMA: y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
                alpha = settings.HAND_GRIP_SMOOTHING_ALPHA
                grip_factor = alpha * grip_factor + (1.0 - alpha) * last_grip
                # 更新状态
                grip_state['value'] = grip_factor

        # 2. 估算 3D 位置 (Camera Space)
        # 使用几何相似三角形估算深度 Z
        # 我们使用两种参考长度进行加权估算，并应用角度校正，以提高鲁棒性
        
        # 0. 准备相机参数
        if camera_matrix is None:
             focal_length = (frame_width / 2.0) / math.tan(math.radians(self.fov) / 2.0)
        else:
             focal_length = camera_matrix[0, 0]

        # 参数 A: 纵向长度 (Wrist to Middle MCP)
        # 真实长度: ~0.09m (9cm)
        REF_LENGTH_UP_M = settings.HAND_REF_LENGTH_M
        
        # 参数 B: 横向宽度 (Index MCP to Pinky MCP)
        # 真实长度: ~0.06m (6.0cm)
        # 通常成年男性手掌宽度(MCP处)约 8-9cm，但 Index(5) 到 Pinky(17) 不包含拇指侧，且是掌骨头间距
        # 约为手掌宽度的 2/3。这里取 0.06m 作为经验值
        REF_LENGTH_ACROSS_M = settings.HAND_REF_WIDTH_M
        
        # 计算 Pitch (绕 X 轴旋转)
        # Pitch = asin(-ny) (假设 normal 已经归一化)
        # 当手掌直立正对时，normal=(0,0,1)，ny=0 -> Pitch=0
        # 当手掌向上仰 (指尖指向相机)，normal 指向 +Y? 不，normal 始终垂直于手掌
        # 如果手掌向后倒 (指尖远离相机)，v_up 偏向 Z。
        pitch = math.asin(np.clip(-normal[1], -1.0, 1.0))
        pitch_deg = math.degrees(pitch)
        
        # 估算 Z (基于 Up 向量 + Pitch 校正)
        # 投影长度 L_proj_up = L_real * cos(Pitch) * (f / Z)
        # Z = (f * L_real * cos(Pitch)) / L_proj_up
        # 注意：这里我们使用 2D 投影长度 (忽略 z 分量)，因为我们要显式进行角度校正
        # 如果使用包含 z 的 3D 长度，理论上不需要 cos(Pitch)，但 MediaPipe Z 可能不准
        # 用户特别强调 "Yaw角校正"，暗示希望显式处理投影关系。
        
        dist_up_2d = np.linalg.norm(p9[:2] - p0[:2]) # 只取 x, y
        
        # 避免除零
        if dist_up_2d < 1e-4:
            z_up_raw = 0.5
        else:
            # 限制校正角度，避免 90 度时 cos 为 0
            # 使用 abs(cos) 处理手背朝向 (Pitch ~ 180) 的情况
            # 当 Pitch ~ 0 (手心正对) 或 180 (手背正对) 时，cos ~ 1，投影最大，估算最准
            # 当 Pitch ~ 90 (指尖对准相机) 时，cos ~ 0，投影最小，估算不准
            cos_pitch = max(0.2, abs(math.cos(pitch))) 
            len_px_up = dist_up_2d * frame_width
            # 计算 Z
            z_up_raw = (focal_length * REF_LENGTH_UP_M * cos_pitch) / len_px_up

        # 估算 Z (基于 Across 向量 + Yaw 校正)
        # 投影长度 W_proj = W_real * cos(Yaw) * (f / Z)
        # Z = (f * W_real * cos(Yaw)) / W_proj
        
        dist_across_2d = np.linalg.norm(p17[:2] - p5[:2]) # 只取 x, y
        
        if dist_across_2d < 1e-4:
            z_across = 0.5
        else:
            # 限制校正角度
            # 同理使用 abs(cos) 处理手背情况 (Yaw 可能接近 180 如果定义反转)
            # 实际上 Yaw 通常在 -90 到 90 之间，但如果手背朝向，Normal 反转可能导致 Yaw 突变
            # 无论 Normal 指向哪里，投影长度只与平面夹角有关，即 abs(dot(view, vec))
            # 简单来说就是 abs(cos(angle))
            cos_yaw = max(0.2, abs(math.cos(yaw)))
            len_px_across = dist_across_2d * frame_width
            z_across = (focal_length * REF_LENGTH_ACROSS_M * cos_yaw) / len_px_across
            
        # --- 动态长度校准 (Dynamic Length Correction) ---
        # 目的：确保 z_up 与 z_across 一致 (以 Width 为基准)，避免通道切换深度跳变
        # 修改方向：反向校准，用 Width 校准 Length
        
        length_correction_factor = 1.0
        if width_correction_state is not None:
            # 复用 state 字典，虽然名字叫 width_correction_state，现在存的是 length correction
            length_correction_factor = width_correction_state.get('value', 1.0)
            
            # 仅在手掌展开、静止且正对时更新校准系数
            # 临时计算 motion_score (仅用于更新校准)
            temp_motion_score = 1.0
            if depth_length_history and len(depth_length_history) >= 2:
                 # 注意：此时 history 还没 push 当前帧
                 sigma = np.std(depth_length_history)
                 avg = np.mean(depth_length_history)
                 if avg > 1e-6:
                     temp_motion_score = min(max(sigma / (avg * settings.HAND_DEPTH_SIGMA_THRESHOLD_RATIO), 0.0), 1.0)
            
            can_update_correction = (temp_motion_score < 0.2) and \
                                    (grip_factor < 0.2) and \
                                    (abs(yaw_deg) < 20.0) and \
                                    (abs(pitch_deg) < 20.0)
            
            if can_update_correction:
                # 计算目标系数: z_across / z_up_raw
                # 目标是 z_up_corrected = z_up_raw * factor ≈ z_across
                # 所以 factor = z_across / z_up_raw
                target_factor = z_across / z_up_raw
                target_factor = max(0.5, min(target_factor, 2.0))
                
                # EMA 更新
                alpha_corr = 0.05 # 慢速更新
                length_correction_factor = alpha_corr * target_factor + (1.0 - alpha_corr) * length_correction_factor
                width_correction_state['value'] = length_correction_factor
        
        # 应用校准
        z_up = z_up_raw * length_correction_factor

        # 融合策略
        # 哪个角度小，哪个维度的投影就更可靠 (cos值大，受噪声影响小)
        # 使用 cos 值作为权重，同样取绝对值
        w_up = max(0.2, abs(math.cos(pitch)))
        w_across = max(0.2, abs(math.cos(yaw)))
        
        # 根据聚拢系数调整权重：握拳时 (grip_factor -> 1)，减少长度通道 (Up) 的权重
        # 目标：握拳时 w_up = 0，展开时 w_up 不变
        w_up *= (1.0 - grip_factor)

        # 归一化权重
        w_sum = w_up + w_across
        z_est = (z_up * w_up + z_across * w_across) / w_sum
        
        # 记录详细调试参数
        depth_details = {
            'z_up': z_up,
            'z_across': z_across,
            'w_up': w_up / w_sum,
            'w_across': w_across / w_sum,
            'len_corr': length_correction_factor # Rename for clarity
        }
        
        # 备选简单策略：如果 Yaw 很大 (>60度)，主要信赖 Up；如果 Pitch 很大，主要信赖 Across
        # 上述加权已隐含此逻辑
            
        # 计算 X, Y (Camera Space)
        # Center of Hand (use Middle MCP 9 or approx center)
        # Let's use Palm Center approx: Midpoint of 0 and 9 is roughly center.
        # Or (0 + 5 + 17)/3.
        # Let's use (0 + 9) / 2
        center_x_norm = (landmarks[0].x + landmarks[9].x) / 2.0
        center_y_norm = (landmarks[0].y + landmarks[9].y) / 2.0
        
        # Convert to pixel coords
        center_x_px = center_x_norm * frame_width
        center_y_px = center_y_norm * frame_height
        
        cx = frame_width / 2.0
        cy = frame_height / 2.0
        if camera_matrix is not None:
            cx = camera_matrix[0, 2]
            cy = camera_matrix[1, 2]
            
        # X = (u - cx) * Z / fx
        x_est = (center_x_px - cx) * z_est / focal_length
        # Y = (v - cy) * Z / fy
        # fy usually equals fx
        y_est = (center_y_px - cy) * z_est / focal_length
        
        # 3. 深度变化率检测 (Motion Score)
        motion_score = 0.0
        if depth_length_history is not None:
            # 添加当前长度通道深度估计值到历史
            depth_length_history.append(z_up)
            if len(depth_length_history) > settings.HAND_DEPTH_HISTORY_SIZE:
                depth_length_history.pop(0)
            
            # 计算标准差
            if len(depth_length_history) >= 2:
                sigma_length = np.std(depth_length_history)
                
                # 计算阈值 (2-3% 的当前深度值，或者历史平均深度值)
                # 使用历史平均值更稳定
                avg_depth = np.mean(depth_length_history)
                sigma_threshold = avg_depth * settings.HAND_DEPTH_SIGMA_THRESHOLD_RATIO
                
                # 避免除零
                if sigma_threshold < 1e-6:
                    sigma_threshold = 1e-6
                    
                # 计算 motion_score
                motion_score = min(max(sigma_length / sigma_threshold, 0.0), 1.0)

        # 3.2 动态噪声 R 计算
        # R = R_base + k * R_max * (1 - motion_score)
        r_base = settings.HAND_KALMAN_R_BASE
        r_grip_max = settings.HAND_KALMAN_R_GRIP_MAX
        
        r_dynamic = r_base + grip_factor * r_grip_max * (1.0 - motion_score)

        # 3.3 深度锚定 (Depth Anchor) 逻辑
        # 维护一个"锚定深度值"，当 Yaw 较小且手掌展开时更新
        # 当握拳时，将锚定值作为额外权重注入深度估计
        
        anchor_weight = 0.0
        anchor_depth = 0.0
        
        if anchor_state is not None:
            # 检查是否可以更新锚定值
            # Yaw 绝对值低于阈值 且 手掌展开 (grip_factor 低)
            can_update_anchor = (abs(yaw_deg) < settings.HAND_DEPTH_ANCHOR_YAW_THRESHOLD) and \
                                (grip_factor < settings.HAND_DEPTH_ANCHOR_GRIP_THRESHOLD)
                                
            if can_update_anchor:
                # 更新锚定值 (使用当前 Across 通道深度，因为它在正面时更准，或者使用融合深度 z_est)
                # 这里使用 z_est，因为它是经过 Yaw/Pitch 校正融合的
                anchor_state['value'] = z_est
                anchor_state['frame_id'] = frame_id
                anchor_state['timestamp'] = timestamp if timestamp else time.time()
                
            # 计算锚定权重并应用融合
            # 只有在握拳状态 (grip_factor > 0) 且有有效锚定值时才应用
            if anchor_state['value'] > 0 and grip_factor > 0:
                # 1. 新鲜度衰减
                frames_since_update = frame_id - anchor_state['frame_id']
                if frames_since_update < 0: frames_since_update = 0
                
                # 半衰期计算 decay = exp(-t / tau)
                # tau = half_life / ln(2)
                half_life = settings.HAND_DEPTH_ANCHOR_HALFLIFE_FRAMES
                decay = math.exp(-frames_since_update * 0.693 / half_life)
                
                # 2. 运动衰减 (静止时权重高)
                motion_factor = 1.0 - motion_score
                
                # 3. 聚拢系数 (握拳越深，越依赖锚定)
                # grip_factor 已经在 0-1 之间
                
                # 综合权重
                w_anchor = decay * motion_factor * grip_factor
                
                # 限制最大权重，避免过度锁定
                w_anchor = min(w_anchor, 0.8) 
                
                anchor_weight = w_anchor
                anchor_depth = anchor_state['value']
                
                # 融合深度
                # z_final = (1 - w) * z_current + w * z_anchor
                z_est = (1.0 - w_anchor) * z_est + w_anchor * anchor_depth

        # 4. 滤波
        
        # Yaw Filter
        if one_euro_filter_dict is not None and timestamp is not None:
            if 'yaw' not in one_euro_filter_dict:
                one_euro_filter_dict['yaw'] = OneEuroFilter(
                    min_cutoff=settings.HAND_YAW_ONE_EURO_MIN_CUTOFF, 
                    beta=settings.HAND_YAW_ONE_EURO_BETA,
                    d_cutoff=settings.HAND_YAW_ONE_EURO_D_CUTOFF
                )
            yaw_deg = one_euro_filter_dict['yaw'].filter(yaw_deg, timestamp)
            
        # Pos Filter
        if pos_filter:
            # 应用动态 R (仅 Z 轴)
            x_est, y_est, z_est = pos_filter.update(x_est, y_est, z_est, R_z=r_dynamic)
            
        # W_norm (for compatibility/visualizer)
        # Just use distance between 5 and 17 in unified space
        w_norm = np.linalg.norm(p17 - p5)
        
        return x_est, y_est, z_est, w_norm, yaw_deg, pitch_deg, motion_score, grip_factor, depth_details

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
                'depth_length_history': [],
                'depth_anchor': {'value': 0.0, 'frame_id': 0, 'timestamp': 0.0},
                'width_correction': {'value': 1.0, 'count': 0},
                'grip_state': {'value': 0.0}
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
                'depth_length_history': [],
                'depth_anchor': {'value': 0.0, 'frame_id': 0, 'timestamp': 0.0},
                'width_correction': {'value': 1.0, 'count': 0},
                'grip_state': {'value': 0.0}
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
                        depth_length_history = None
                        anchor_state = None
                        grip_state = None
                        width_correction_state = None
                        label = "Unknown"
                        
                        if result_lite.multi_handedness and idx < len(result_lite.multi_handedness):
                            # handedness[0] is the category with highest score
                            categories = result_lite.multi_handedness[idx]
                            if categories:
                                label = categories[0]['label'] # "Left" or "Right"
                                if label in self.hand_filters:
                                    w_norm_filter = self.hand_filters[label]['w_norm']
                                    pos_filter = self.hand_filters[label]['pos']
                                    depth_length_history = self.hand_filters[label]['depth_length_history']
                                    anchor_state = self.hand_filters[label]['depth_anchor']
                                    grip_state = self.hand_filters[label]['grip_state']
                                    width_correction_state = self.hand_filters[label]['width_correction']
                                    # 检查是否有 OneEuroFilter 字典用于关键点滤波
                                    if 'landmarks' not in self.hand_filters[label]:
                                        self.hand_filters[label]['landmarks'] = {}
                                    one_euro_filter_dict = self.hand_filters[label]['landmarks']
                        
                        x, y, z, w_norm, yaw, pitch, motion_score, grip_factor, depth_details = self._calculate_hand_pos(
                            landmarks, target_w, target_h, aspect_ratio, 
                            w_norm_filter=w_norm_filter, 
                            pos_filter=pos_filter, 
                            one_euro_filter_dict=one_euro_filter_dict,
                            timestamp=timestamp_ms / 1000.0,
                            camera_matrix=cached_camera_matrix,
                            hand_label=label,
                            depth_length_history=depth_length_history,
                            anchor_state=anchor_state,
                            grip_state=grip_state,
                            width_correction_state=width_correction_state,
                            frame_id=frame_id
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
        detector.close()
        for mgr in self.shm_managers:
            try:
                mgr.close()
            except:
                pass
