
import cv2
import numpy as np
# import mediapipe as mp # 移除 mediapipe 导入，避免 AttributeError
from config.settings import LEFT_EYE_CENTER_MODEL, RIGHT_EYE_CENTER_MODEL, EYE_RADIUS, AXIS_LENGTH
from utils.math_utils import calculate_single_eye_gaze, calculate_screen_intersection, calculate_weighted_average

class Visualizer:
    def __init__(self):
        # 初始化可视化缓存数据 (用于消除闪烁)
        self.cached_viz_data = {
            'l_start': None, 'l_end': None, 
            'r_start': None, 'r_end': None,
            'text': None, 'text_color': (255, 0, 255),
            'l_eye_center': None, 'r_eye_center': None
        }
        # 定义手部连接关系
        self.HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]

    def render(self, frame, roi_info, eye_points, raw_eye_points, tracker, fps, gaze_data=None, hand_result=None, drop_rate=0.0):
        """
        统一渲染入口
        """
        # 1. 绘制 ROI
        self._draw_roi(frame, roi_info)

        # 2. 绘制手部关键点
        if hand_result:
            self.draw_hands(frame, hand_result)

        # 3. 绘制虹膜
        if eye_points and len(eye_points) == 2:
            self._draw_iris(frame, eye_points, raw_eye_points)
            
            # 绘制瞳孔间距线
            p1 = (int(eye_points[0][0]), int(eye_points[0][1]))
            p2 = (int(eye_points[1][0]), int(eye_points[1][1]))
            cv2.line(frame, p1, p2, (255, 255, 0), 2)

        # 3. 更新并绘制视线 (如果有数据)
        if gaze_data:
            self._update_gaze_viz(
                gaze_data['rvec'], 
                gaze_data['tvec'], 
                eye_points, 
                gaze_data['cam_matrix'], 
                gaze_data['dist_coeffs']
            )

        # 4. 绘制所有覆盖信息 (Info, Gaze Lines, Crosshair)
        self._draw_overlay(frame, tracker, fps, drop_rate)

        # 5. 显示并处理按键
        cv2.imshow('Face and Eye Detection (MediaPipe)', frame)
        return cv2.waitKey(1) & 0xFF == 27 # Returns True if ESC pressed

    def draw_hands(self, frame, hand_result):
        """
        绘制手部关键点
        """
        if not hand_result or not hand_result.multi_hand_landmarks:
            return

        h, w, _ = frame.shape
        
        for hand_landmarks_lite in hand_result.multi_hand_landmarks:
             # Draw connections
            for connection in self.HAND_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                
                # 确保索引不越界
                if start_idx < len(hand_landmarks_lite) and end_idx < len(hand_landmarks_lite):
                    start_point = hand_landmarks_lite[start_idx]
                    end_point = hand_landmarks_lite[end_idx]
                    
                    x1, y1 = int(start_point.x * w), int(start_point.y * h)
                    x2, y2 = int(end_point.x * w), int(end_point.y * h)
                    
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
            # Draw landmarks
            for landmark in hand_landmarks_lite:
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
                cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)

    def _draw_roi(self, frame, roi_info):
         h, w = frame.shape[:2]
         rx, ry, rw, rh, _ = roi_info
         if rw < w or rh < h:
             cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (255, 0, 0), 1)

    def _draw_iris(self, frame, eye_points, raw_eye_points):
         # 绘制虹膜中心 (使用滤波后的坐标绘制，以反馈真实追踪位置)
         f_p1, f_p2 = eye_points
         cv2.circle(frame, (int(f_p1[0]), int(f_p1[1])), 3, (0, 255, 0), -1, cv2.LINE_AA) # Green for filtered
         cv2.circle(frame, (int(f_p2[0]), int(f_p2[1])), 3, (0, 255, 0), -1, cv2.LINE_AA)
         
         # 绘制原始点作为对比 (红色)
         if raw_eye_points:
             cx_left, cy_left = raw_eye_points[0]
             cx_right, cy_right = raw_eye_points[1]
             cv2.circle(frame, (cx_left, cy_left), 2, (0, 0, 255), -1, cv2.LINE_AA)
             cv2.circle(frame, (cx_right, cy_right), 2, (0, 0, 255), -1, cv2.LINE_AA)

    def _update_gaze_viz(self, rvec, tvec, eye_points, cam_matrix, dist_coeffs):
        # 获取虹膜 2D 坐标
        if len(eye_points) < 2:
            return
            
        left_iris_2d = eye_points[0]
        right_iris_2d = eye_points[1]
        
        # 计算左右眼视线
        l_gaze_vec, l_eye_center_cam = calculate_single_eye_gaze(
            left_iris_2d, LEFT_EYE_CENTER_MODEL, rvec, tvec, cam_matrix, dist_coeffs, eye_radius=EYE_RADIUS
        )
        r_gaze_vec, r_eye_center_cam = calculate_single_eye_gaze(
            right_iris_2d, RIGHT_EYE_CENTER_MODEL, rvec, tvec, cam_matrix, dist_coeffs, eye_radius=EYE_RADIUS
        )
        
        # --- 计算视线与屏幕平面 (Z=0) 的交点 ---
        l_screen_point = calculate_screen_intersection(l_eye_center_cam, l_gaze_vec)
        r_screen_point = calculate_screen_intersection(r_eye_center_cam, r_gaze_vec)
        
        # --- 加权平均 ---
        avg_screen_point = calculate_weighted_average(l_screen_point, r_screen_point)
        
        # --- 准备绘制数据 ---
        # 调整：视线起点改为虹膜位置 (眼球表面)
        l_start_3d = l_eye_center_cam + l_gaze_vec
        r_start_3d = r_eye_center_cam + r_gaze_vec
        
        # 终点从虹膜再延伸
        l_end_3d = l_eye_center_cam + l_gaze_vec * (AXIS_LENGTH / 60.0)
        r_end_3d = r_eye_center_cam + r_gaze_vec * (AXIS_LENGTH / 60.0)

        # 投影关键点
        points_to_project = np.array([l_start_3d, l_end_3d, r_start_3d, r_end_3d])
        projected_points, _ = cv2.projectPoints(points_to_project, np.zeros((3,1)), np.zeros((3,1)), cam_matrix, dist_coeffs)
        
        # 更新缓存
        self.cached_viz_data['l_start'] = (int(projected_points[0][0][0]), int(projected_points[0][0][1]))
        self.cached_viz_data['l_end'] = (int(projected_points[1][0][0]), int(projected_points[1][0][1]))
        self.cached_viz_data['r_start'] = (int(projected_points[2][0][0]), int(projected_points[2][0][1]))
        self.cached_viz_data['r_end'] = (int(projected_points[3][0][0]), int(projected_points[3][0][1]))
        
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

    def _draw_overlay(self, frame, tracker, fps, drop_rate=0.0):
        h, w = frame.shape[:2]
        
        # 绘制视线向量和交点信息 (使用缓存的数据，消除闪烁)
        if self.cached_viz_data['text'] is not None:
            # 绘制视线向量
            if self.cached_viz_data['l_start'] is not None:
                cv2.line(frame, self.cached_viz_data['l_start'], self.cached_viz_data['l_end'], (255, 0, 0), 2)
                cv2.line(frame, self.cached_viz_data['r_start'], self.cached_viz_data['r_end'], (255, 0, 0), 2)

            # 绘制交点文本
            cv2.putText(frame, self.cached_viz_data['text'], (10, 85), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.cached_viz_data['text_color'], 2)

        # 绘制光轴中心（十字准星）
        center_x, center_y = w // 2, h // 2
        cv2.line(frame, (center_x - 10, center_y), (center_x + 10, center_y), (0, 0, 255), 1)
        cv2.line(frame, (center_x, center_y - 10), (center_x, center_y + 10), (0, 0, 255), 1)

        # 绘制 FPS (左上角第一行) 和 丢包率
        pd_display = f" | PD: {int(tracker.current_pixel_dist)}px" if tracker.current_pixel_dist > 0 else ""
        
        # 格式化丢包率显示
        drop_color = (0, 255, 0) # Green for low drop
        if drop_rate > 0.1: drop_color = (0, 255, 255) # Yellow
        if drop_rate > 0.3: drop_color = (0, 0, 255) # Red
        
        info_text = f"FPS: {int(fps)} | Drop: {drop_rate*100:.1f}%{pd_display}"
        cv2.putText(frame, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, drop_color, 2)
        
        # 绘制主视眼标记（黄色圆圈 + 'R'）
        if tracker.dominant_eye_pos is not None:
            dx, dy = tracker.dominant_eye_pos
            if 0 <= dx < w and 0 <= dy < h:
                cv2.circle(frame, (dx, dy), 8, (0, 255, 255), 2)
                cv2.putText(frame, "R", (dx + 10, dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if tracker.current_pixel_dist > 0:
            if tracker.current_estimated_dist > 200:
                info_text = "too far!"
                cv2.putText(frame, info_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                # 重构为空间坐标显示
                # 参考系：原点摄像头，Z光轴，X水平，Y竖直
                head_pos_text = f"Head Pos: X:{int(tracker.current_offset_x)} Y:{int(tracker.current_offset_y)} Z:{int(tracker.current_estimated_dist)} cm"
                cv2.putText(frame, head_pos_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
