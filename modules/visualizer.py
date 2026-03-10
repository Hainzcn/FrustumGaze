
import cv2
import numpy as np
# import mediapipe as mp # 移除 mediapipe 导入，避免 AttributeError
from config.settings import LEFT_EYE_CENTER_MODEL, RIGHT_EYE_CENTER_MODEL, EYE_RADIUS, AXIS_LENGTH, GAZE_RENDER_INTERVAL
from utils.math_utils import calculate_screen_intersection, calculate_weighted_average

class Visualizer:
    def __init__(self):
        # 初始化可视化缓存数据 (用于消除闪烁)
        self.cached_viz_data = {
            'l_start': None, 'l_end': None, 
            'r_start': None, 'r_end': None,
            'text': None, 'text_color': (255, 0, 255),
            'l_eye_center': None, 'r_eye_center': None
        }
        self.render_counter = 0 # 渲染计数器，用于控制频率
        
        # 字体与绘制参数
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX
        self.FONT_SCALE_INFO = 0.6
        self.FONT_SCALE_TEXT = 0.5
        self.FONT_THICKNESS = 2
        
        # 定义手部连接关系
        self.HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]

    def render(self, frame, roi_info, eye_points, raw_eye_points, tracker, fps, gaze_data=None, hand_result=None, pose_result=None, drop_rate=0.0, p99_latency=0.0, hands_pos=None, closest_hand=None, using_full_scan=False):
        """
        统一渲染入口
        """
        # 0. 绘制 ROI 选框 (调试用)
        if roi_info:
            roi_x, roi_y, roi_w, roi_h, _ = roi_info
            # 全图扫描时用红色虚线/实线，ROI 模式用绿色实线
            color = (0, 0, 255) if using_full_scan else (0, 255, 0)
            thickness = 2
            cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), color, thickness)
            
            # 标注模式
            mode_text = "FULL SCAN - SEARCHING" if using_full_scan else "ROI MODE - TRACKING"
            cv2.putText(frame, mode_text, (roi_x, max(20, roi_y - 10)), self.FONT, 0.5, color, 1)

        # 1. 绘制手部和姿态关键点
        # 即使 hand_result 为 None，如果 pose_result 有值也应该绘制
        if hand_result or pose_result:
            # 获取头部距离用于过滤手部
            head_dist_cm = tracker.current_estimated_dist if tracker else 0.0
            self.draw_hands(frame, hand_result, pose_result, hands_pos, closest_hand, head_dist_cm=head_dist_cm)
        
        # 如果是全图扫描模式，仅绘制 ROI 框，不绘制后续的面部细节
        if using_full_scan:
            # 5. 显示并处理按键 (保持显示逻辑)
            cv2.imshow('Face and Eye Detection (MediaPipe)', frame)
            return cv2.waitKey(1) & 0xFF == 27 

        # 3. 绘制虹膜
        if eye_points and len(eye_points) == 2:
            self._draw_iris(frame, eye_points, raw_eye_points)
            
            # 已移除瞳孔间距线绘制 (cv2.line)

        # 3. 更新并绘制视线 (如果有数据)
        # 根据用户指令：视线线段渲染跟随视线解算相同频率
        # 因此，只有当 main 传递了有效的 gaze_data 时，才更新渲染数据
        if gaze_data:
            self._update_gaze_viz_with_tracker(
                gaze_data['rvec'], 
                gaze_data['tvec'], 
                eye_points, 
                gaze_data['cam_matrix'], 
                gaze_data['dist_coeffs'],
                tracker,
                rmat=gaze_data.get('rmat')
            )
        else:
            # 如果没有新数据（非解算帧），清空或保持？
            # 根据 "全图扫描时不渲染视线线段"，这里其实已经在上方 if using_full_scan 中被拦截了
            # 但对于 ROI 模式下的非解算帧，gaze_data 为 None
            # 如果不更新，_draw_overlay 将使用 cached_viz_data 绘制上一帧的线
            # 这通常是期望的（画面流畅），但如果用户想要明确的"不渲染"，则需要清空 cache
            # 鉴于 "视线线段渲染跟随视线解算相同频率"，这可能意味着线段的 *位置更新* 跟随解算频率，而不是 *闪烁*
            # 但为了严谨，如果 gaze_data 为 None (非解算帧)，我们不调用 update，_draw_overlay 会继续画旧的
            pass
        
        # 4. 绘制所有覆盖信息 (Info, Gaze Lines, Crosshair)
        self._draw_overlay(frame, tracker, fps, drop_rate, p99_latency)

        # 5. 显示并处理按键
        cv2.imshow('Face and Eye Detection (MediaPipe)', frame)
        return cv2.waitKey(1) & 0xFF == 27 # Returns True if ESC pressed

    def draw_hands(self, frame, hand_result, pose_result=None, hands_pos=None, closest_hand=None, head_dist_cm=0.0):
        """
        绘制手部关键点 (已优化)
        """
        h, w = frame.shape[:2]

        # Draw Pose Landmarks (Shoulders, Elbows)
        # 11: Left Shoulder, 12: Right Shoulder
        # 13: Left Elbow, 14: Right Elbow
        # PoseTracker extracts 11, 12, 13, 14 (in this order)
        
        pose_px = []
        left_elbow_px = None
        right_elbow_px = None
        
        # Visibility threshold for pose landmarks
        VISIBILITY_THRESHOLD = 0.5

        if pose_result and pose_result.pose_landmarks:
            pose_lms = pose_result.pose_landmarks
            # Convert to pixels
            for lm in pose_lms:
                 pose_px.append((int(lm.x * w), int(lm.y * h)))
            
            if len(pose_px) >= 4:
                # Define indices for clarity based on PoseTracker output
                # 0: Left Shoulder, 1: Right Shoulder
                # 2: Left Elbow, 3: Right Elbow
                idx_l_sh, idx_r_sh = 0, 1
                idx_l_el, idx_r_el = 2, 3
                
                l_shoulder_px = pose_px[idx_l_sh]
                r_shoulder_px = pose_px[idx_r_sh]
                l_elbow_px = pose_px[idx_l_el]
                r_elbow_px = pose_px[idx_r_el]
                
                # Check visibility
                l_sh_vis = getattr(pose_lms[idx_l_sh], 'visibility', 1.0)
                r_sh_vis = getattr(pose_lms[idx_r_sh], 'visibility', 1.0)
                l_el_vis = getattr(pose_lms[idx_l_el], 'visibility', 1.0)
                r_el_vis = getattr(pose_lms[idx_r_el], 'visibility', 1.0)
                
                # Draw Shoulders Connection (Only if both visible)
                if l_sh_vis > VISIBILITY_THRESHOLD and r_sh_vis > VISIBILITY_THRESHOLD:
                    cv2.line(frame, l_shoulder_px, r_shoulder_px, (255, 255, 0), 2)
                
                # Draw Left Arm (Shoulder -> Elbow)
                if l_sh_vis > VISIBILITY_THRESHOLD and l_el_vis > VISIBILITY_THRESHOLD:
                    cv2.line(frame, l_shoulder_px, l_elbow_px, (255, 255, 0), 2)
                    left_elbow_px = l_elbow_px # Assign only if visible
                
                # Draw Right Arm (Shoulder -> Elbow)
                if r_sh_vis > VISIBILITY_THRESHOLD and r_el_vis > VISIBILITY_THRESHOLD:
                    cv2.line(frame, r_shoulder_px, r_elbow_px, (255, 255, 0), 2)
                    right_elbow_px = r_elbow_px # Assign only if visible
                
                # Draw Points (Only visible ones)
                for i, px in enumerate(pose_px):
                    vis = getattr(pose_lms[i], 'visibility', 1.0)
                    if vis > VISIBILITY_THRESHOLD:
                        cv2.circle(frame, px, 5, (255, 0, 0), -1) # Blue

        if not hand_result or not hand_result.multi_hand_landmarks:
            return
        
        # 优化: 预处理 hands_pos 为字典 O(1) 查找
        hands_pos_map = {}
        if hands_pos:
            hands_pos_map = {p['id']: p for p in hands_pos}
        
        for idx, hand_landmarks_lite in enumerate(hand_result.multi_hand_landmarks):
            # 获取该手的捏起状态
            is_pinching = False
            pinch_pos = (0,0,0)
            pinch_center_2d = (0,0)
            hand_label = "Unknown"
            
            # 优化: 直接字典查找
            hand_pos = hands_pos_map.get(idx)
            if hand_pos:
                # 过滤逻辑: 如果手部距离大于头部距离 + 阈值，则跳过绘制
                # head_dist_cm 为头部到摄像头的距离 (cm)
                # hand_pos['z'] 为手部到摄像头的距离 (m)，需要转换
                # 阈值设为 10cm，容忍误差
                hand_z_cm = hand_pos['z'] * 100
                if head_dist_cm > 0 and hand_z_cm > (head_dist_cm + 10.0):
                    continue
                
                is_pinching = hand_pos.get('is_pinching', False)
                pinch_pos = hand_pos.get('pinch_pos', (0,0,0))
                pinch_center_2d = hand_pos.get('pinch_center_2d', (0,0))
                hand_label = hand_pos.get('label', "Unknown")

            # 优化: 预计算所有关键点像素坐标，避免重复 float->int 转换和乘法
            landmarks_px = []
            for lm in hand_landmarks_lite:
                landmarks_px.append((int(lm.x * w), int(lm.y * h)))

            # Draw Connection from Elbow to Wrist (Index 0)
            wrist_px = landmarks_px[0]
            if hand_label == "Left" and left_elbow_px:
                cv2.line(frame, left_elbow_px, wrist_px, (255, 255, 0), 2)
            elif hand_label == "Right" and right_elbow_px:
                cv2.line(frame, right_elbow_px, wrist_px, (255, 255, 0), 2)

            # Draw connections
            for connection in self.HAND_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                
                # 确保索引不越界
                if start_idx < len(landmarks_px) and end_idx < len(landmarks_px):
                    start_point = landmarks_px[start_idx]
                    end_point = landmarks_px[end_idx]
                    
                    # 确定颜色
                    color = (0, 255, 0) # 默认绿色
                    if is_pinching:
                        color = (0, 0, 255) # 捏起时为红色
                    elif closest_hand and closest_hand['id'] == idx:
                        color = (0, 165, 255) # 最近的手为橙色 (如果不捏起)
                        
                    cv2.line(frame, start_point, end_point, color, 2)
                
            # Draw landmarks
            for px_point in landmarks_px:
                point_color = (0, 0, 255) # 默认点外圈红色
                if is_pinching:
                    point_color = (0, 0, 255) # 捏起时保持红色
                
                cv2.circle(frame, px_point, 4, point_color, -1)
                cv2.circle(frame, px_point, 2, (255, 255, 255), -1)
            
            # Draw 3D Position Text
            if hand_pos:
                # 在手腕位置显示坐标
                wx, wy = landmarks_px[0]
                
                # 格式化: X, Y, Z (cm)
                # 注意：我们计算的是 meters, 转换为 cm
                # 计算像素距离 (PD) = w_norm * frame_width (近似)
                pd_val = hand_pos.get('w_norm', 0) * w
                text = f"PD:{pd_val:.0f}px X:{hand_pos['x']*100:.0f} Y:{hand_pos['y']*100:.0f} Z:{hand_pos['z']*100:.0f}cm"
                
                text_color = (0, 255, 0)
                if is_pinching:
                    text_color = (0, 0, 255)
                elif closest_hand and closest_hand['id'] == idx:
                    text_color = (0, 165, 255)
                    text += " (Closest)"
                
                # 优化: 使用常量字体参数
                cv2.putText(frame, text, (wx, wy + 20), self.FONT, self.FONT_SCALE_TEXT, text_color, self.FONT_THICKNESS)
                
                # 显示 Yaw 和 Pitch 角
                yaw_val = hand_pos.get('yaw', 0.0)
                pitch_val = hand_pos.get('pitch', 0.0)
                angle_text = f"Yaw:{yaw_val:.0f} Pitch:{pitch_val:.0f}"
                cv2.putText(frame, angle_text, (wx, wy + 40), self.FONT, self.FONT_SCALE_TEXT, text_color, self.FONT_THICKNESS)
                
                # 绘制详细深度信息
                depth_details = hand_pos.get('depth_details', {})
                if depth_details:
                    z_up = depth_details.get('z_up', 0.0)
                    z_across = depth_details.get('z_across', 0.0)
                    w_up = depth_details.get('w_up', 0.0)
                    w_across = depth_details.get('w_across', 0.0)
                    # Use 'len_corr' instead of 'width_corr' to reflect the change
                    len_corr = depth_details.get('len_corr', 1.0)
                    
                    detail_text_1 = f"Z_UP:{z_up:.2f}m (W:{w_up:.2f})"
                    detail_text_2 = f"Z_AC:{z_across:.2f}m (W:{w_across:.2f})"
                    detail_text_3 = f"L-Corr:{len_corr:.2f}"
                    
                    cv2.putText(frame, detail_text_1, (wx, wy + 60), self.FONT, 0.4, (200, 200, 200), 1)
                    cv2.putText(frame, detail_text_2, (wx, wy + 75), self.FONT, 0.4, (200, 200, 200), 1)
                    cv2.putText(frame, detail_text_3, (wx, wy + 90), self.FONT, 0.4, (200, 200, 200), 1)
                
                # 如果捏起，显示指尖位置信息 (已取消空间坐标显示，只保留2D标记)
                if is_pinching:
                    # 绘制捏起点半透明圆
                    cx, cy = pinch_center_2d
                    if cx > 0 and cy > 0:
                        p_x, p_y = int(cx * w), int(cy * h)
                        radius = 15
                        
                        # 优化: 使用 ROI 混合替代全帧拷贝
                        # 计算 ROI 边界，注意不要越界
                        x1 = max(0, p_x - radius)
                        y1 = max(0, p_y - radius)
                        x2 = min(w, p_x + radius)
                        y2 = min(h, p_y + radius)
                        
                        # 只有当 ROI 有效时才绘制
                        if x2 > x1 and y2 > y1:
                            roi = frame[y1:y2, x1:x2]
                            overlay = roi.copy()
                            
                            # 在 ROI 局部坐标系中绘制
                            # 圆心在 ROI 中的坐标
                            local_center = (p_x - x1, p_y - y1)
                            cv2.circle(overlay, local_center, radius, (255, 0, 255), -1) # 紫色实心圆
                            
                            # 混合并写回原图
                            cv2.addWeighted(overlay, 0.5, roi, 0.5, 0, roi)
                            frame[y1:y2, x1:x2] = roi

    def _draw_iris(self, frame, eye_points, raw_eye_points):
        # 绘制虹膜中心 (使用滤波后的坐标绘制，以反馈真实追踪位置)
        f_p1, f_p2 = eye_points
        cv2.circle(frame, (int(f_p1[0]), int(f_p1[1])), 3, (0, 255, 0), -1) # Green for filtered
        cv2.circle(frame, (int(f_p2[0]), int(f_p2[1])), 3, (0, 255, 0), -1)
        
        # 绘制原始点作为对比 (红色)
        if raw_eye_points:
            cx_left, cy_left = raw_eye_points[0]
            cx_right, cy_right = raw_eye_points[1]
            cv2.circle(frame, (int(cx_left), int(cy_left)), 2, (0, 0, 255), -1)
            cv2.circle(frame, (int(cx_right), int(cy_right)), 2, (0, 0, 255), -1)

    def _update_gaze_viz_with_tracker(self, rvec, tvec, eye_points, cam_matrix, dist_coeffs, tracker, rmat=None):
        # 获取虹膜 2D 坐标
        if len(eye_points) < 2:
            return
            
        left_iris_2d = eye_points[0]
        right_iris_2d = eye_points[1]
        
        # 计算左右眼视线
        l_gaze_vec, l_eye_center_cam = tracker.calculate_single_eye_gaze(
            left_iris_2d, LEFT_EYE_CENTER_MODEL, rvec, tvec, cam_matrix, dist_coeffs, eye_radius=EYE_RADIUS, rmat=rmat
        )
        r_gaze_vec, r_eye_center_cam = tracker.calculate_single_eye_gaze(
            right_iris_2d, RIGHT_EYE_CENTER_MODEL, rvec, tvec, cam_matrix, dist_coeffs, eye_radius=EYE_RADIUS, rmat=rmat
        )
        
        # --- 计算视线与屏幕平面 (Z=0) 的交点 ---
        l_screen_point = calculate_screen_intersection(l_eye_center_cam, l_gaze_vec)
        r_screen_point = calculate_screen_intersection(r_eye_center_cam, r_gaze_vec)
        
        # --- 加权平均 ---
        avg_screen_point = calculate_weighted_average(l_screen_point, r_screen_point)
        
        # --- 准备绘制数据 ---
        # 调整：视线起点改为虹膜位置 (2D 投影点)
        # 我们已经有虹膜的 2D 坐标 (eye_points)，可以直接使用
        # 但为了保持与 l_end_3d 的 3D 一致性，我们最好投影 l_start_3d
        # l_start_3d = l_eye_center_cam + l_gaze_vec * (EYE_RADIUS) # 实际上是虹膜在球面的位置
        
        # 计算虹膜在球面的 3D 位置
        # calculate_single_eye_gaze 返回的 gaze_vector 是从 eye_center 指向虹膜的
        # 且其长度已被归一化为 EYE_RADIUS (如果相交)
        
        # 终点从虹膜再延伸
        # 为了视觉效果，起点设为虹膜位置
        l_start_3d = l_eye_center_cam + l_gaze_vec
        r_start_3d = r_eye_center_cam + r_gaze_vec
        
        l_end_3d = l_start_3d + l_gaze_vec * (AXIS_LENGTH / 60.0) # 延伸长度
        r_end_3d = r_start_3d + r_gaze_vec * (AXIS_LENGTH / 60.0)

        # 投影关键点
        points_to_project = np.array([l_start_3d, l_end_3d, r_start_3d, r_end_3d])
        projected_points, _ = cv2.projectPoints(points_to_project, np.zeros((3,1)), np.zeros((3,1)), cam_matrix, dist_coeffs)
        
        # 优化：使用检测到的 2D 虹膜坐标作为起点，消除投影误差
        # 计算投影向量 (方向和长度)
        # l_vec = p_end - p_start
        # r_vec = p_end - p_start
        
        l_p_start = projected_points[0][0]
        l_p_end = projected_points[1][0]
        r_p_start = projected_points[2][0]
        r_p_end = projected_points[3][0]
        
        l_vec = l_p_end - l_p_start
        r_vec = r_p_end - r_p_start
        
        # 使用真实 landmark 坐标作为起点
        l_real_start = np.array(eye_points[0])
        r_real_start = np.array(eye_points[1])
        
        # 计算新的终点
        l_final_end = l_real_start + l_vec
        r_final_end = r_real_start + r_vec
        
        # 更新缓存
        # 确保坐标是整数
        self.cached_viz_data['l_start'] = (int(l_real_start[0]), int(l_real_start[1]))
        self.cached_viz_data['l_end'] = (int(l_final_end[0]), int(l_final_end[1]))
        self.cached_viz_data['r_start'] = (int(r_real_start[0]), int(r_real_start[1]))
        self.cached_viz_data['r_end'] = (int(r_final_end[0]), int(r_final_end[1]))
        
        # 更新交点文本缓存
        if avg_screen_point is not None:
            ix_cm = avg_screen_point[0] / 50.0
            iy_cm = avg_screen_point[1] / 50.0
            # Z is always 0
            
            self.cached_viz_data['text'] = f"Gaze on Screen: X:{int(ix_cm)} Y:{int(iy_cm)} cm"
            self.cached_viz_data['text_color'] = (255, 0, 255)
        else:
            self.cached_viz_data['text'] = "Gaze on Screen: N/A"
            self.cached_viz_data['text_color'] = (0, 0, 255)

    def _draw_overlay(self, frame, tracker, fps, drop_rate=0.0, p99_latency=0.0):
        h, w = frame.shape[:2]
        
        # 绘制视线向量和交点信息 (使用缓存的数据，消除闪烁)
        if self.cached_viz_data['text'] is not None:
            # 绘制视线向量
            if self.cached_viz_data['l_start'] is not None:
                cv2.line(frame, self.cached_viz_data['l_start'], self.cached_viz_data['l_end'], (255, 0, 0), 2)
                cv2.line(frame, self.cached_viz_data['r_start'], self.cached_viz_data['r_end'], (255, 0, 0), 2)

        # 绘制光轴中心（十字准星）
        center_x, center_y = w // 2, h // 2
        cv2.line(frame, (center_x - 10, center_y), (center_x + 10, center_y), (0, 0, 255), 1)
        cv2.line(frame, (center_x, center_y - 10), (center_x, center_y + 10), (0, 0, 255), 1)

        # 绘制 FPS (左上角第一行) 和 丢包率
        # 格式化丢包率显示
        drop_color = (0, 255, 0) # Green for low drop
        if drop_rate > 0.1: drop_color = (0, 255, 255) # Yellow
        if drop_rate > 0.3: drop_color = (0, 0, 255) # Red
        
        info_text = f"FPS: {int(fps)} | Drop: {drop_rate*100:.1f}% | P99: {int(p99_latency)}ms"
        cv2.putText(frame, info_text, (10, 25), self.FONT, self.FONT_SCALE_INFO, drop_color, self.FONT_THICKNESS)
        
        # 绘制头部位置信息
        if tracker.current_pixel_dist > 0:
            if tracker.current_estimated_dist > 200:
                head_text = "Too far!"
            else:
                head_text = f"PD: {int(tracker.current_pixel_dist)}px | Head: X:{int(tracker.current_offset_x)} Y:{int(tracker.current_offset_y)} Z:{int(tracker.current_estimated_dist)} cm"
            cv2.putText(frame, head_text, (10, 50), self.FONT, self.FONT_SCALE_TEXT, (0, 255, 255), self.FONT_THICKNESS)
            
            # 显示 Head Yaw
            # Tracker 中应该存储了 yaw
            if hasattr(tracker, 'current_yaw'):
                yaw_text = f"Head Yaw: {tracker.current_yaw:.1f} deg"
                cv2.putText(frame, yaw_text, (10, 70), self.FONT, self.FONT_SCALE_TEXT, (0, 255, 255), self.FONT_THICKNESS)

        # 绘制视线交点信息
        if self.cached_viz_data['text'] is not None:
            gaze_text = self.cached_viz_data['text']
            color = self.cached_viz_data['text_color']
            cv2.putText(frame, gaze_text, (10, 90), self.FONT, self.FONT_SCALE_TEXT, color, self.FONT_THICKNESS)

        # 绘制头部中心标记（黄色圆圈 + 'C'）
        if tracker.head_center_pos is not None:
            dx, dy = tracker.head_center_pos
            if 0 <= dx < w and 0 <= dy < h:
                # 修改为实心点 (radius=4, thickness=-1)
                cv2.circle(frame, (dx, dy), 4, (0, 255, 255), -1)
                cv2.putText(frame, "C", (dx + 10, dy), self.FONT, self.FONT_SCALE_TEXT, (0, 255, 255), 1)
