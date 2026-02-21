
import cv2
import time
import math
import numpy as np
from modules.filters import OneEuroFilter, OneDKalmanFilter
from config.settings import MODEL_POINTS
import config.settings as settings

class EyeTracker:
    def __init__(self):
        self.current_pixel_dist = 0
        self.current_estimated_dist = 0
        self.current_offset_x = 0
        self.current_offset_y = 0
        
        # 初始化距离平滑卡尔曼滤波器（二级层）
        self.pixel_dist_filter = OneDKalmanFilter(Q=settings.FACE_DIST_KALMAN_Q, R=settings.FACE_DIST_KALMAN_R)
        self.real_dist_filter = OneDKalmanFilter(Q=settings.FACE_DIST_KALMAN_Q, R=settings.FACE_DIST_KALMAN_R)
        # 偏移量滤波器（二级层）
        self.offset_x_filter = OneDKalmanFilter(Q=settings.FACE_OFFSET_KALMAN_Q, R=settings.FACE_OFFSET_KALMAN_R)
        self.offset_y_filter = OneDKalmanFilter(Q=settings.FACE_OFFSET_KALMAN_Q, R=settings.FACE_OFFSET_KALMAN_R)
        
        # 记录主视眼位置用于绘制
        self.dominant_eye_pos = None

        # 初始化 OneEuroFilter 字典
        self.filters = {}
        self.filters_initialized = False

    def reset(self):
        self.dominant_eye_pos = None
        self.filters = {}
        self.filters_initialized = False
        # 保持当前距离值直到计算出新值以避免闪烁

    def _get_filter(self, name, value, min_cutoff=settings.FACE_ONE_EURO_MIN_CUTOFF, beta=settings.FACE_ONE_EURO_BETA):
        current_time = time.time()
        if name not in self.filters:
            self.filters[name] = OneEuroFilter(current_time, value, min_cutoff=min_cutoff, beta=beta, d_cutoff=settings.FACE_ONE_EURO_D_CUTOFF)
            return value
        return self.filters[name].filter(current_time, value)

    def filter_eye_points(self, eye_points):
        """
        对原始像素坐标进行滤波
        eye_points: [(lx, ly), (rx, ry)]
        """
        if len(eye_points) < 2:
            return eye_points
            
        lx, ly = eye_points[0]
        rx, ry = eye_points[1]

        f_lx = self._get_filter('lx', lx)
        f_ly = self._get_filter('ly', ly)
        f_rx = self._get_filter('rx', rx)
        f_ry = self._get_filter('ry', ry)
        
        self.filters_initialized = True
        
        return [(f_lx, f_ly), (f_rx, f_ry)]

    def apply_distance_filter(self, pixel_dist, estimated_dist):
        """应用Kalman滤波处理距离数据"""
        # 数据有效性检查
        if pixel_dist <= 0 or estimated_dist <= 0:
            return self.current_pixel_dist, self.current_estimated_dist
            
        filtered_pixel = self.pixel_dist_filter.update(pixel_dist)
        filtered_estimated = self.real_dist_filter.update(estimated_dist)
        
        return filtered_pixel, filtered_estimated

    def _extract_landmark_point(self, landmarks, idx, w, h):
        """Extracts a specific landmark point and converts to pixel coordinates."""
        try:
            point = landmarks[idx]
            return np.array([point.x * w, point.y * h])
        except (IndexError, AttributeError):
            return None

    def process_landmarks(self, face_landmarks, frame_width, frame_height, camera_fov, cam_matrix, dist_coeffs):
        """
        Process face landmarks to extract eye points, calculate distance and head pose.
        Returns a dictionary with results.
        """
        w, h = frame_width, frame_height
        
        # 1. Extract and Filter Points of Interest
        # Iris
        iris_l = self._extract_landmark_point(face_landmarks, 468, w, h)
        iris_r = self._extract_landmark_point(face_landmarks, 473, w, h)
        
        # Inner Eye Corners (133, 362)
        inner_l = self._extract_landmark_point(face_landmarks, 133, w, h)
        inner_r = self._extract_landmark_point(face_landmarks, 362, w, h)
        
        # Outer Eye Corners (33, 263)
        outer_l = self._extract_landmark_point(face_landmarks, 33, w, h)
        outer_r = self._extract_landmark_point(face_landmarks, 263, w, h)
        
        if any(p is None for p in [iris_l, iris_r, inner_l, inner_r, outer_l, outer_r]):
            return None

        # Apply OneEuro Filter to all points
        # Using specific keys for each coordinate
        f_iris_l = np.array([self._get_filter('iris_lx', iris_l[0]), self._get_filter('iris_ly', iris_l[1])])
        f_iris_r = np.array([self._get_filter('iris_rx', iris_r[0]), self._get_filter('iris_ry', iris_r[1])])
        
        f_inner_l = np.array([self._get_filter('inner_lx', inner_l[0]), self._get_filter('inner_ly', inner_l[1])])
        f_inner_r = np.array([self._get_filter('inner_rx', inner_r[0]), self._get_filter('inner_ry', inner_r[1])])
        
        f_outer_l = np.array([self._get_filter('outer_lx', outer_l[0]), self._get_filter('outer_ly', outer_l[1])])
        f_outer_r = np.array([self._get_filter('outer_rx', outer_r[0]), self._get_filter('outer_ry', outer_r[1])])
        
        # Format for return and legacy support
        eye_points = [(f_iris_l[0], f_iris_l[1]), (f_iris_r[0], f_iris_r[1])]
        raw_eye_points = [(iris_l[0], iris_l[1]), (iris_r[0], iris_r[1])]

        # 2. Calculate Distance (using filtered inner/outer eye corners)
        d_inner_px = np.linalg.norm(f_inner_l - f_inner_r)
        d_outer_px = np.linalg.norm(f_outer_l - f_outer_r)
        
        # Real distances (cm)
        D_INNER_REAL = 4.0
        D_OUTER_REAL = 9.0
        
        # Focal length calculation
        fov_rad = math.radians(camera_fov)
        focal_length = (w / 2) / math.tan(fov_rad / 2)
        
        # Estimate depth
        z_inner = (D_INNER_REAL * focal_length) / d_inner_px if d_inner_px > 0 else 0
        z_outer = (D_OUTER_REAL * focal_length) / d_outer_px if d_outer_px > 0 else 0
        
        if z_inner > 0 and z_outer > 0:
            estimated_dist = (z_inner + z_outer) / 2.0
        elif z_inner > 0:
            estimated_dist = z_inner
        else:
            estimated_dist = z_outer
            
        # Representative pixel distance (for filtering compatibility)
        pixel_dist = (6.5 * focal_length) / estimated_dist if estimated_dist > 0 else 0

        # 3. Calculate Head Pose (PnP)
        # Using raw points for PnP as it usually benefits from raw data, 
        # but we could use filtered points if we filtered all mesh points (too expensive).
        # We'll use the specific PnP logic here.
        pitch, yaw, roll, rvec, tvec = self._calculate_head_pose(face_landmarks, w, h, cam_matrix, dist_coeffs)
        
        # 4. Apply correction and filtering
        correction_factor = math.cos(math.radians(yaw))
        if correction_factor < 0.2: correction_factor = 0.2
        
        corrected_dist = estimated_dist * correction_factor
        
        filtered_pixel, filtered_estimated = self.apply_distance_filter(pixel_dist, corrected_dist)
        self.current_pixel_dist = filtered_pixel
        self.current_estimated_dist = filtered_estimated
        
        # 5. Update offset
        self.update_offset(eye_points, w, h, filtered_pixel, filtered_estimated, camera_fov)
        
        return {
            'eye_points': eye_points,
            'raw_eye_points': raw_eye_points,
            'rvec': rvec,
            'tvec': tvec,
            'yaw': yaw,
            'pitch': pitch,
            'roll': roll
        }

    def _calculate_head_pose(self, face_landmarks, w, h, cam_matrix, dist_coeffs):
        """Calculates head pose using PnP."""
        # 2D 图像点 (使用 MediaPipe 关键点索引)
        # 1: Nose Tip, 152: Chin, 33: Left Eye Outer, 263: Right Eye Outer, 
        # 61: Left Mouth Corner, 291: Right Mouth Corner, 133: Left Eye Inner, 
        # 362: Right Eye Inner, 70: Left Eyebrow Outer, 300: Right Eyebrow Outer, 2: Nose Bottom
        indices = [1, 152, 33, 263, 61, 291, 133, 362, 70, 300, 2]
        
        image_points = []
        for idx in indices:
            pt = self._extract_landmark_point(face_landmarks, idx, w, h)
            if pt is None:
                return 0, 0, 0, None, None
            image_points.append(pt)
            
        image_points = np.array(image_points, dtype="double")
        
        # PnP 求解
        (success, rotation_vector, translation_vector) = cv2.solvePnP(MODEL_POINTS, image_points, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
    
        if not success:
            return 0, 0, 0, None, None
    
        # 计算欧拉角
        rmat, jac = cv2.Rodrigues(rotation_vector)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
        
        # angles[0]=pitch, angles[1]=yaw, angles[2]=roll
        return angles[0], angles[1], angles[2], rotation_vector, translation_vector

    def calculate_single_eye_gaze(self, iris_center_2d, eye_center_model_3d, rvec, tvec, cam_matrix, dist_coeffs, eye_radius=60.0):
        """
        计算单眼视线向量
        :param iris_center_2d: (x, y) 像素坐标
        :param eye_center_model_3d: (x, y, z) 模型坐标系下的眼球中心
        :param rvec: 头部旋转向量
        :param tvec: 头部平移向量
        :param cam_matrix: 相机内参
        :return: (gaze_vector_3d, eye_center_cam_3d) 相机坐标系下的视线向量和眼球中心
        """
        # 1. 将眼球中心变换到相机坐标系
        rmat, _ = cv2.Rodrigues(rvec)
        eye_center_cam = np.dot(rmat, eye_center_model_3d) + tvec.reshape(3)
        
        # 2. 将虹膜 2D 点反投影为射线 (相机坐标系)
        # 先去畸变？为简单起见假设畸变很小或直接使用原始坐标
        # 射线方向: inv(K) * [u, v, 1]
        p_iris_h = np.array([iris_center_2d[0], iris_center_2d[1], 1.0])
        ray_dir = np.dot(np.linalg.inv(cam_matrix), p_iris_h)
        ray_dir = ray_dir / np.linalg.norm(ray_dir) # Normalize
        
        # 3. 计算射线与眼球球面的交点 (或者近似)
        # 球体：圆心 = eye_center_cam，半径 = 60 (12mm * 50 units/cm)
        radius = eye_radius
        
        # 射线: O = (0,0,0), D = ray_dir
        # 交点: |t*D - C|^2 = r^2
        # t^2 - 2(C.D)t + |C|^2 - r^2 = 0
        C = eye_center_cam
        a = 1.0
        b = -2.0 * np.dot(C, ray_dir)
        c_val = np.dot(C, C) - radius**2
        
        discriminant = b**2 - 4*a*c_val
        
        if discriminant >= 0:
            # 射线与球体相交
            t1 = (-b - math.sqrt(discriminant)) / (2*a)
            t2 = (-b + math.sqrt(discriminant)) / (2*a)
            t = min(t1, t2) if t1 > 0 and t2 > 0 else max(t1, t2) # Should be positive
            
            if t > 0:
                iris_on_sphere = t * ray_dir
                gaze_vector = iris_on_sphere - eye_center_cam
                return gaze_vector, eye_center_cam
                
        # Fallback: 如果没有交点（计算误差），使用射线上的最近点
        # 射线上距离 C 最近的点在 t = C.D 处
        t_closest = np.dot(C, ray_dir)
        closest_point = t_closest * ray_dir
        # 强制长度为半径
        vec = closest_point - eye_center_cam
        vec_norm = np.linalg.norm(vec)
        if vec_norm > 0:
            gaze_vector = vec / vec_norm * radius
        else:
            # 回退到头部前方方向 (相机空间中的头部 Z 轴)
            # 相机空间中的头部 Z 轴是 R 的第 3 列
            gaze_vector = rmat[:, 2] * radius
            
        return gaze_vector, eye_center_cam


    def update_offset(self, eye_points, frame_width, frame_height, pixel_dist, real_dist_cm, fov=60.0):
        """
        计算并更新右眼（画面左侧）相对于摄像机光轴的物理偏移
        采用针孔成像模型 (Pinhole Camera Model):
        X = Z * (x - cx) / fx
        Y = Z * (y - cy) / fy
        """
        # 如果距离无效，尝试使用缓存
        if real_dist_cm <= 0:
            if self.current_estimated_dist > 0:
                real_dist_cm = self.current_estimated_dist
            else:
                return

        if len(eye_points) == 0:
            return
            
        # 1. 确定右眼坐标 (u, v)
        # 假设 eye_points 中 x 坐标较小的是右眼（画面左侧）
        sorted_eyes = sorted(eye_points, key=lambda p: p[0])
        right_eye = sorted_eyes[0]
        self.dominant_eye_pos = (int(right_eye[0]), int(right_eye[1]))
        u, v = right_eye
        
        # 2. 确定相机内参 (Intrinsics)
        # 主点 (Principal Point) 坐标 (cx, cy)
        # 严格定义为图像中心，消除系统性偏差
        cx = frame_width / 2.0
        cy = frame_height / 2.0
        
        # 焦距 (Focal Length) fx, fy
        # 假设像素是正方形 (square pixels)，即 fx = fy
        # 使用与 calculate_distance 一致的 FOV = 60度 计算焦距
        fov_rad = math.radians(fov)
        # tan(fov/2) = (w/2) / f  =>  f = (w/2) / tan(fov/2)
        f = (frame_width / 2.0) / math.tan(fov_rad / 2.0)
        fx = f
        fy = f
        
        # 3. 获取深度 Z (cm)
        Z = real_dist_cm
        
        # 4. 应用针孔模型公式计算物理偏移 (X, Y)
        # x轴向右为正，y轴向下为正 (符合图像坐标系)
        # 当 u = cx, v = cy 时，结果严格为 0
        real_offset_x = Z * (u - cx) / fx
        real_offset_y = Z * (v - cy) / fy
        
        # 滤波 (Keep Secondary Filter)
        filtered_x = self.offset_x_filter.update(real_offset_x)
        filtered_y = self.offset_y_filter.update(real_offset_y)
        
        # 保留浮点数精度
        self.current_offset_x = filtered_x
        self.current_offset_y = filtered_y
