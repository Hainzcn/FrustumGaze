
"""
可视化渲染模块。
负责将追踪结果（人脸、眼动、手部、姿态）实时绘制在视频帧上，并显示系统统计信息。
"""

import cv2
import numpy as np
from config.settings import EYE_RADIUS, AXIS_LENGTH, GAZE_RENDER_INTERVAL

class Visualizer:
    """
    可视化类，管理所有绘制逻辑。
    提供统一的渲染入口，并缓存部分数据以减少画面闪烁。
    """
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
        
        # 定义手部关键点连接关系 (MediaPipe 21个关键点的标准连接)
        self.HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),      # 大拇指
            (0, 5), (5, 6), (6, 7), (7, 8),      # 食指
            (5, 9), (9, 10), (10, 11), (11, 12), # 中指
            (9, 13), (13, 14), (14, 15), (15, 16), # 无名指
            (13, 17), (17, 18), (18, 19), (19, 20), # 小拇指
            (0, 17)                              # 掌根
        ]

    def render(self, frame, 
               roi_info=None, 
               eye_points=None, 
               raw_eye_points=None, 
               gaze_result=None, 
               fps=0.0, 
               gaze_data=None,
               hand_result=None,
               pose_result=None,
               drop_rate=0.0,
               p99_latency=0.0,
               hands_pos=None,
               closest_hand=None,
               using_full_scan=False):
        """
        渲染可视化内容。
        统一入口，负责调用各子模块绘制 ROI、手部、姿态、虹膜、视线及统计信息。
        注意：waitKey 只能在主线程调用一次，返回值包含按键信息。
        返回: (bool) 是否按下 ESC 退出键
        """
        if frame is None:
            return self.check_exit_key()

        # 1. 绘制 ROI 区域
        self._draw_roi(frame, roi_info, using_full_scan)

        # 2. 绘制手部和姿态 (包含手臂连接)
        head_dist = gaze_result.estimated_dist if gaze_result else 0.0
        self.draw_hands(frame, hand_result, pose_result, hands_pos, closest_hand, head_dist_cm=head_dist)
        
        # 3. 绘制虹膜中心点
        if eye_points:
             self._draw_iris(frame, eye_points, raw_eye_points)

        # 4. 更新并绘制视线向量
        if gaze_data and gaze_result and eye_points:
            if 'cam_matrix' in gaze_data and gaze_data['cam_matrix'] is not None:
                self._update_gaze_viz_with_tracker(
                    gaze_data['rvec'], 
                    gaze_data['tvec'], 
                    eye_points, 
                    gaze_data['cam_matrix'], 
                    gaze_data['dist_coeffs'], 
                    gaze_result, 
                    rmat=gaze_data.get('rmat')
                )
            
        # 5. 绘制叠加层信息 (FPS, 丢包率, 头部位置文本等)
        if gaze_result:
             self._draw_overlay(frame, gaze_result, fps, drop_rate, p99_latency)

        # 6. 显示最终画面
        cv2.imshow('FrustumGaze', frame)
        return self.check_exit_key()

    def _draw_roi(self, frame, roi_info, using_full_scan):
        """绘制当前追踪的 ROI 区域。"""
        if not roi_info:
            return
            
        roi_x, roi_y, roi_w, roi_h, _ = roi_info
        # 全图扫描使用红色标识搜索，ROI 模式使用绿色标识锁定
        color = (0, 0, 255) if using_full_scan else (0, 255, 0)
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), color, 2)
        
        mode_text = "FULL SCAN - SEARCHING" if using_full_scan else "ROI MODE - TRACKING"
        cv2.putText(frame, mode_text, (roi_x, max(20, roi_y - 10)), self.FONT, 0.5, color, 1)

    def draw_hands(self, frame, hand_result, pose_result=None, hands_pos=None, closest_hand=None, head_dist_cm=0.0):
        """
        绘制手部关键点和肢体姿态。
        包含手臂连接、手部骨架、捏合状态标记以及 3D 位置信息。
        """
        h, w = frame.shape[:2]
        
        # 1. 绘制躯干与手臂姿态 (Shoulders, Elbows)
        left_elbow_px, right_elbow_px = self._draw_pose_landmarks(frame, pose_result, w, h)

        # 2. 绘制手部关键点
        if not hand_result or not hand_result.multi_hand_landmarks:
            return
        
        # 建立手部 ID 到位置数据的映射，加速查找
        hands_pos_map = {p['id']: p for p in hands_pos} if hands_pos else {}
        
        for idx, hand_landmarks_lite in enumerate(hand_result.multi_hand_landmarks):
            hand_pos = hands_pos_map.get(idx)
            if not hand_pos:
                continue
                
            hand_z_cm = hand_pos['z']
            if head_dist_cm > 0 and hand_z_cm > (head_dist_cm + 10.0):
                continue

            # 提取该手的状态
            is_pinching = hand_pos.get('is_pinching', False)
            pinch_center_2d = hand_pos.get('pinch_center_2d', (0,0))
            hand_label = hand_pos.get('label', "Unknown")

            # 预计算关键点像素坐标
            landmarks_px = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks_lite]

            # 绘制手臂末端连接 (肘部 -> 腕部)
            wrist_px = landmarks_px[0]
            if hand_label == "Left" and left_elbow_px:
                cv2.line(frame, left_elbow_px, wrist_px, (255, 255, 0), 2)
            elif hand_label == "Right" and right_elbow_px:
                cv2.line(frame, right_elbow_px, wrist_px, (255, 255, 0), 2)

            # 确定绘制颜色 (捏起时红色，最近的手橙色，普通绿色)
            color = (0, 255, 0)
            if is_pinching:
                color = (0, 0, 255)
            elif closest_hand and closest_hand['id'] == idx:
                color = (0, 165, 255)

            # 绘制手部骨架连接线
            for connection in self.HAND_CONNECTIONS:
                if connection[0] < len(landmarks_px) and connection[1] < len(landmarks_px):
                    cv2.line(frame, landmarks_px[connection[0]], landmarks_px[connection[1]], color, 2)
                
            # 绘制关键点节点
            for px_point in landmarks_px:
                cv2.circle(frame, px_point, 4, (0, 0, 255), -1)
                cv2.circle(frame, px_point, 2, (255, 255, 255), -1)
            
            # 绘制坐标文本信息
            self._draw_hand_info_text(frame, landmarks_px[0], hand_pos, is_pinching, closest_hand, idx, color)
            
            # 绘制捏合点高亮标记
            if is_pinching:
                self._draw_pinch_marker(frame, pinch_center_2d, w, h)

    def _draw_pose_landmarks(self, frame, pose_result, w, h):
        """绘制躯干和手臂姿态，返回左右肘部的像素坐标。"""
        left_elbow_px = None
        right_elbow_px = None
        VISIBILITY_THRESHOLD = 0.5

        if not pose_result or not pose_result.pose_landmarks:
            return None, None

        pose_lms = pose_result.pose_landmarks
        pose_px = [(int(lm.x * w), int(lm.y * h)) for lm in pose_lms]
        
        if len(pose_px) >= 4:
            # 索引定义：0:左肩, 1:右肩, 2:左肘, 3:右肘
            l_sh_vis = getattr(pose_lms[0], 'visibility', 1.0)
            r_sh_vis = getattr(pose_lms[1], 'visibility', 1.0)
            l_el_vis = getattr(pose_lms[2], 'visibility', 1.0)
            r_el_vis = getattr(pose_lms[3], 'visibility', 1.0)
            
            # 绘制肩膀连线
            if l_sh_vis > VISIBILITY_THRESHOLD and r_sh_vis > VISIBILITY_THRESHOLD:
                cv2.line(frame, pose_px[0], pose_px[1], (255, 255, 0), 2)
            
            # 绘制左臂
            if l_sh_vis > VISIBILITY_THRESHOLD and l_el_vis > VISIBILITY_THRESHOLD:
                cv2.line(frame, pose_px[0], pose_px[2], (255, 255, 0), 2)
                left_elbow_px = pose_px[2]
            
            # 绘制右臂
            if r_sh_vis > VISIBILITY_THRESHOLD and r_el_vis > VISIBILITY_THRESHOLD:
                cv2.line(frame, pose_px[1], pose_px[3], (255, 255, 0), 2)
                right_elbow_px = pose_px[3]
            
            # 绘制关节圆点
            for i, px in enumerate(pose_px):
                if getattr(pose_lms[i], 'visibility', 1.0) > VISIBILITY_THRESHOLD:
                    cv2.circle(frame, px, 5, (255, 0, 0), -1)

        return left_elbow_px, right_elbow_px

    def _draw_hand_info_text(self, frame, wrist_px, hand_pos, is_pinching, closest_hand, idx, color):
        """在手部位置显示 3D 坐标、角度及深度解算详情。"""
        wx, wy = wrist_px
        
        # 1. 基础坐标与状态文本
        pd_val = hand_pos.get('w_norm', 0) * frame.shape[1]
        text = f"PD:{pd_val:.0f}px X:{hand_pos['x']:.0f} Y:{hand_pos['y']:.0f} Z:{hand_pos['z']:.0f}cm"
        if closest_hand and closest_hand['id'] == idx:
            text += " (Closest)"
        cv2.putText(frame, text, (wx, wy + 20), self.FONT, self.FONT_SCALE_TEXT, color, self.FONT_THICKNESS)
        
        # 2. 偏航/俯仰角文本
        angle_text = f"Yaw:{hand_pos.get('yaw', 0.0):.0f} Pitch:{hand_pos.get('pitch', 0.0):.0f}"
        cv2.putText(frame, angle_text, (wx, wy + 40), self.FONT, self.FONT_SCALE_TEXT, color, self.FONT_THICKNESS)
        
        # 3. 深度融合详情 (调试用小字)
        depth_details = hand_pos.get('depth_details', {})
        if depth_details:
            detail_text_1 = f"Z_UP:{depth_details.get('z_up', 0.0):.1f}cm (W:{depth_details.get('w_up', 0.0):.2f})"
            detail_text_2 = f"Z_AC:{depth_details.get('z_across', 0.0):.1f}cm (W:{depth_details.get('w_across', 0.0):.2f})"
            detail_text_3 = f"L-Corr:{depth_details.get('len_corr', 1.0):.2f}"
            
            for i, d_text in enumerate([detail_text_1, detail_text_2, detail_text_3]):
                cv2.putText(frame, d_text, (wx, wy + 60 + i*15), self.FONT, 0.4, (200, 200, 200), 1)

    def _draw_pinch_marker(self, frame, pinch_center_2d, w, h):
        """绘制捏合动作的视觉反馈（紫色半透明标记）。"""
        cx, cy = pinch_center_2d
        if cx <= 0 or cy <= 0:
            return
            
        p_x, p_y = int(cx * w), int(cy * h)
        radius = 15
        
        # 计算 ROI 边界，防止越界
        x1, y1 = max(0, p_x - radius), max(0, p_y - radius)
        x2, y2 = min(w, p_x + radius), min(h, p_y + radius)
        
        if x2 > x1 and y2 > y1:
            roi = frame[y1:y2, x1:x2]
            overlay = roi.copy()
            cv2.circle(overlay, (p_x - x1, p_y - y1), radius, (255, 0, 255), -1)
            cv2.addWeighted(overlay, 0.5, roi, 0.5, 0, roi)
            frame[y1:y2, x1:x2] = roi

    def _draw_iris(self, frame, eye_points, raw_eye_points):
        """绘制虹膜中心点（包含滤波后点与原始点对比）。"""
        # 1. 绘制滤波后的虹膜中心 (绿色)
        f_p1, f_p2 = eye_points
        cv2.circle(frame, (int(f_p1[0]), int(f_p1[1])), 3, (0, 255, 0), -1)
        cv2.circle(frame, (int(f_p2[0]), int(f_p2[1])), 3, (0, 255, 0), -1)
        
        # 2. 绘制原始检测点 (红色)，用于直观观察滤波延迟/平滑效果
        if raw_eye_points:
            cx_left, cy_left = raw_eye_points[0]
            cx_right, cy_right = raw_eye_points[1]
            cv2.circle(frame, (int(cx_left), int(cy_left)), 2, (0, 0, 255), -1)
            cv2.circle(frame, (int(cx_right), int(cy_right)), 2, (0, 0, 255), -1)

    @staticmethod
    def _draw_transparent_line(frame, pt1, pt2, color, thickness, alpha):
        """在 frame 上绘制半透明线段，仅对线段 bounding box 区域做 overlay 混合。"""
        h, w = frame.shape[:2]
        pad = thickness + 1
        x1 = max(0, min(pt1[0], pt2[0]) - pad)
        y1 = max(0, min(pt1[1], pt2[1]) - pad)
        x2 = min(w, max(pt1[0], pt2[0]) + pad)
        y2 = min(h, max(pt1[1], pt2[1]) + pad)
        if x2 <= x1 or y2 <= y1:
            return
        roi = frame[y1:y2, x1:x2]
        overlay = roi.copy()
        offset = (-x1, -y1)
        cv2.line(overlay, (pt1[0] + offset[0], pt1[1] + offset[1]),
                 (pt2[0] + offset[0], pt2[1] + offset[1]), color, thickness)
        cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)

    def _update_gaze_viz_with_tracker(self, rvec, tvec, eye_points, cam_matrix, dist_coeffs, gaze_result, rmat=None):
        """
        使用 GazeResult 中预计算的视线向量更新可视化缓存数据。
        通过 3D->2D 透视投影保留视线方向的纵深感。
        """
        if len(eye_points) < 2:
            return

        l_gaze_vec = gaze_result.left_gaze_vec
        r_gaze_vec = gaze_result.right_gaze_vec
        l_eye_center_cam = gaze_result.left_eye_center_cam
        r_eye_center_cam = gaze_result.right_eye_center_cam

        if l_gaze_vec is None or r_gaze_vec is None:
            return

        l_start_3d = l_eye_center_cam + l_gaze_vec * EYE_RADIUS
        r_start_3d = r_eye_center_cam + r_gaze_vec * EYE_RADIUS
        l_end_3d = l_start_3d + l_gaze_vec * AXIS_LENGTH
        r_end_3d = r_start_3d + r_gaze_vec * AXIS_LENGTH

        points_to_project = np.array([l_start_3d, l_end_3d, r_start_3d, r_end_3d])
        projected_points, _ = cv2.projectPoints(
            points_to_project, np.zeros((3, 1)), np.zeros((3, 1)), cam_matrix, dist_coeffs
        )

        l_p_start, l_p_end = projected_points[0][0], projected_points[1][0]
        r_p_start, r_p_end = projected_points[2][0], projected_points[3][0]

        GAZE_MIN_PX, GAZE_MAX_PX = 20, 120

        l_real_start = np.array(eye_points[0])
        r_real_start = np.array(eye_points[1])

        l_dir = l_p_end - l_p_start
        r_dir = r_p_end - r_p_start
        l_mag = np.linalg.norm(l_dir)
        r_mag = np.linalg.norm(r_dir)

        if l_mag > 0:
            clamped_l = max(GAZE_MIN_PX, min(GAZE_MAX_PX, l_mag))
            l_dir = l_dir / l_mag * clamped_l
        if r_mag > 0:
            clamped_r = max(GAZE_MIN_PX, min(GAZE_MAX_PX, r_mag))
            r_dir = r_dir / r_mag * clamped_r

        l_final_end = l_real_start + l_dir
        r_final_end = r_real_start + r_dir

        self.cached_viz_data['l_start'] = (int(l_real_start[0]), int(l_real_start[1]))
        self.cached_viz_data['l_end'] = (int(l_final_end[0]), int(l_final_end[1]))
        self.cached_viz_data['r_start'] = (int(r_real_start[0]), int(r_real_start[1]))
        self.cached_viz_data['r_end'] = (int(r_final_end[0]), int(r_final_end[1]))

        sp = gaze_result.screen_point
        lc = gaze_result.left_confidence
        rc = gaze_result.right_confidence
        if sp is not None:
            self.cached_viz_data['text'] = f"Gaze: X:{sp[0]:.1f} Y:{sp[1]:.1f} cm | Conf L:{lc:.0%} R:{rc:.0%}"
            self.cached_viz_data['text_color'] = (255, 0, 255)
        else:
            self.cached_viz_data['text'] = "Gaze on Screen: N/A"
            self.cached_viz_data['text_color'] = (0, 0, 255)

    def _draw_overlay(self, frame, gaze_result, fps, drop_rate=0.0, p99_latency=0.0):
        """在画面上叠加绘制统计信息、视线线段和中心参考准星。"""
        h, w = frame.shape[:2]
        
        # 1. 绘制视线向量 (半透明蓝色线段, ROI 级 overlay 混合)
        if self.cached_viz_data['l_start'] is not None:
            self._draw_transparent_line(frame, self.cached_viz_data['l_start'], self.cached_viz_data['l_end'], (255, 0, 0), 2, 0.45)
            self._draw_transparent_line(frame, self.cached_viz_data['r_start'], self.cached_viz_data['r_end'], (255, 0, 0), 2, 0.45)

        # 2. 绘制光轴中心十字准星
        center_x, center_y = w // 2, h // 2
        cv2.line(frame, (center_x - 10, center_y), (center_x + 10, center_y), (0, 0, 255), 1)
        cv2.line(frame, (center_x, center_y - 10), (center_x, center_y + 10), (0, 0, 255), 1)

        # 3. 绘制系统状态信息 (FPS, 丢包率, 延迟)
        drop_color = (0, 255, 0)
        if drop_rate > 0.1: drop_color = (0, 255, 255)
        if drop_rate > 0.3: drop_color = (0, 0, 255)
        
        info_text = f"FPS: {int(fps)} | Drop: {drop_rate*100:.1f}% | P99: {int(p99_latency)}ms"
        cv2.putText(frame, info_text, (10, 25), self.FONT, self.FONT_SCALE_INFO, drop_color, self.FONT_THICKNESS)
        
        # 4. 绘制面部/头部追踪详情
        if gaze_result.pixel_dist > 0:
            if gaze_result.estimated_dist > 200:
                head_text = "Too far!"
            else:
                head_text = f"PD: {int(gaze_result.pixel_dist)}px | Head: X:{int(gaze_result.offset_x)} Y:{int(gaze_result.offset_y)} Z:{int(gaze_result.estimated_dist)} cm"
            cv2.putText(frame, head_text, (10, 50), self.FONT, self.FONT_SCALE_TEXT, (0, 255, 255), self.FONT_THICKNESS)
            
            yaw_text = f"Head: Yaw {gaze_result.yaw:.1f} | Pitch {gaze_result.pitch:.1f} deg"
            cv2.putText(frame, yaw_text, (10, 70), self.FONT, self.FONT_SCALE_TEXT, (0, 255, 0), self.FONT_THICKNESS)

            if gaze_result.depth_details:
                d = gaze_result.depth_details
                dual_text = f"ZW:{int(d.get('z_width',0))}({d.get('w_width',0):.2f}) ZL:{int(d.get('z_length',0))}({d.get('w_length',0):.2f}) CW:{d.get('calibrated_width',0):.1f}cm"
                cv2.putText(frame, dual_text, (10, 90), self.FONT, self.FONT_SCALE_TEXT, (200, 200, 200), self.FONT_THICKNESS)

        # 5. 绘制视线交点坐标文本
        if self.cached_viz_data['text'] is not None:
            cv2.putText(frame, self.cached_viz_data['text'], (10, 110), self.FONT, self.FONT_SCALE_TEXT, self.cached_viz_data['text_color'], self.FONT_THICKNESS)

        # 6. 绘制头部中心位置标记 (C)
        if gaze_result.head_center_pos is not None:
            dx, dy = gaze_result.head_center_pos
            if 0 <= dx < w and 0 <= dy < h:
                cv2.circle(frame, (dx, dy), 4, (0, 255, 255), -1)
                cv2.putText(frame, "C", (dx + 10, dy), self.FONT, self.FONT_SCALE_TEXT, (0, 255, 255), 1)

    def check_exit_key(self):
        """
        检查是否按下了退出键 (ESC)。
        注意：cv2.waitKey(1) 会消耗至少 1ms。
        """
        key = cv2.waitKey(1) & 0xFF
        return key == 27 # ESC key
