
import cv2
import time
import math
import numpy as np
from utils.math_utils import OneEuroFilter, OneDKalmanFilter
from config.settings import MODEL_POINTS
import config.settings as settings

class EyeTracker:
    def __init__(self):
        self.current_pixel_dist = 0
        self.current_estimated_dist = 0
        self.current_offset_x = 0
        self.current_offset_y = 0
        
        # 记录 Yaw/Pitch 以便可视化
        self.current_yaw = 0.0
        self.current_pitch = 0.0
        
        # 滤波器配置
        face_config = settings.FILTER_CONFIG['FACE']
        dist_config = face_config['DISTANCE']
        offset_config = face_config['OFFSET']

        # 初始化距离平滑卡尔曼滤波器（二级层）
        self.pixel_dist_filter = OneDKalmanFilter(Q=dist_config['process_noise'], R=dist_config['measurement_noise'])
        self.real_dist_filter = OneDKalmanFilter(Q=dist_config['process_noise'], R=dist_config['measurement_noise'])
        # 偏移量滤波器（二级层）
        self.offset_x_filter = OneDKalmanFilter(Q=offset_config['process_noise'], R=offset_config['measurement_noise'])
        self.offset_y_filter = OneDKalmanFilter(Q=offset_config['process_noise'], R=offset_config['measurement_noise'])
        
        # 记录头部中心位置用于绘制
        self.head_center_pos = None

        # 双通道深度估算初始化
        self.calibrated_width_cm = settings.FACE_REF_WIDTH_CM
        self.calibration_config = settings.FILTER_CONFIG['FACE']['CALIBRATION']
        
        # 深度调试信息
        self.current_depth_details = {}

        # 初始化 OneEuroFilter 字典
        self.filters = {}
        self.filters_initialized = False

    def reset(self):
        self.head_center_pos = None
        self.filters = {}
        self.filters_initialized = False
        self.current_yaw = 0.0
        self.current_pitch = 0.0
        # 保持当前距离值直到计算出新值以避免闪烁

    def _get_filter(self, name, value, current_time, min_cutoff=None, beta=None, d_cutoff=None):
        # 优化：使用 current_time 参数，避免重复调用 time.time()
        # 优化：减少字典查找开销
        f = self.filters.get(name)
        
        # 如果未指定参数，使用默认的关键点滤波配置
        if min_cutoff is None:
            kp_config = settings.FILTER_CONFIG['KEYPOINT']
            min_cutoff = kp_config['min_cutoff']
            beta = kp_config['beta']
            d_cutoff = kp_config['d_cutoff']
            
        if f is None:
            f = OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
            self.filters[name] = f
        return f.filter(value, current_time)

    def filter_eye_points(self, eye_points):
        """
        对原始像素坐标进行滤波
        eye_points: [(lx, ly), (rx, ry)]
        """
        if len(eye_points) < 2:
            return eye_points
            
        lx, ly = eye_points[0]
        rx, ry = eye_points[1]

        current_time = time.time()

        f_lx = self._get_filter('lx', lx, current_time)
        f_ly = self._get_filter('ly', ly, current_time)
        f_rx = self._get_filter('rx', rx, current_time)
        f_ry = self._get_filter('ry', ry, current_time)
        
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
        # 优化：移除 try-except (假设 MediaPipe 输出结构稳定)
        # 优化：返回 tuple 而非 np.array，减少内存分配
        if idx < len(landmarks):
            point = landmarks[idx]
            return (point.x * w, point.y * h)
        return None

    def process_landmarks(self, face_landmarks, frame_width, frame_height, camera_fov, cam_matrix, dist_coeffs, should_calc_gaze=True):
        """
        Process face landmarks to extract eye points, calculate distance and head pose.
        Returns a dictionary with results.
        should_calc_gaze: 如果为 False，则跳过虹膜提取和视线计算，仅更新头部位置
        """
        w, h = frame_width, frame_height
        current_time = time.time() # 优化：一次获取时间戳
        
        # 1. Extract and Filter Points of Interest
        # Iris (仅在需要计算视线时提取)
        iris_l = None
        iris_r = None
        f_iris_l = None
        f_iris_r = None
        
        if should_calc_gaze:
            iris_l = self._extract_landmark_point(face_landmarks, 468, w, h)
            iris_r = self._extract_landmark_point(face_landmarks, 473, w, h)
            
            if iris_l and iris_r:
                iris_config = settings.FILTER_CONFIG['FACE']['IRIS']
                f_iris_l = (
                    self._get_filter('iris_lx', iris_l[0], current_time, min_cutoff=iris_config['min_cutoff'], beta=iris_config['beta'], d_cutoff=iris_config['d_cutoff']), 
                    self._get_filter('iris_ly', iris_l[1], current_time, min_cutoff=iris_config['min_cutoff'], beta=iris_config['beta'], d_cutoff=iris_config['d_cutoff'])
                )
                f_iris_r = (
                    self._get_filter('iris_rx', iris_r[0], current_time, min_cutoff=iris_config['min_cutoff'], beta=iris_config['beta'], d_cutoff=iris_config['d_cutoff']), 
                    self._get_filter('iris_ry', iris_r[1], current_time, min_cutoff=iris_config['min_cutoff'], beta=iris_config['beta'], d_cutoff=iris_config['d_cutoff'])
                )
        
        # Inner Eye Corners (133, 362)
        inner_l = self._extract_landmark_point(face_landmarks, 133, w, h)
        inner_r = self._extract_landmark_point(face_landmarks, 362, w, h)
        
        # Outer Eye Corners (33, 263)
        outer_l = self._extract_landmark_point(face_landmarks, 33, w, h)
        outer_r = self._extract_landmark_point(face_landmarks, 263, w, h)
        
        # 必须检测到眼角才能计算头部位置
        if any(p is None for p in [inner_l, inner_r, outer_l, outer_r]):
            return None

        # Apply OneEuro Filter to all points
        # Using specific keys for each coordinate
        f_inner_l = (self._get_filter('inner_lx', inner_l[0], current_time), self._get_filter('inner_ly', inner_l[1], current_time))
        f_inner_r = (self._get_filter('inner_rx', inner_r[0], current_time), self._get_filter('inner_ry', inner_r[1], current_time))
        
        f_outer_l = (self._get_filter('outer_lx', outer_l[0], current_time), self._get_filter('outer_ly', outer_l[1], current_time))
        f_outer_r = (self._get_filter('outer_rx', outer_r[0], current_time), self._get_filter('outer_ry', outer_r[1], current_time))
        
        # Format for return and legacy support
        eye_points = []
        raw_eye_points = []
        
        if should_calc_gaze and f_iris_l and f_iris_r:
            eye_points = [f_iris_l, f_iris_r]
            raw_eye_points = [iris_l, iris_r]
        else:
            # 如果不计算视线，这里返回空列表，避免下游代码错误使用
            eye_points = [] 
            raw_eye_points = []

        # 2. Calculate Distance (Dual Channel Strategy)
        # --------------------------------------------------------------------------------
        # 通道 A: 宽度通道 (基于外眼角间距) - 易受 Yaw 影响，受 Pitch 影响小
        # 通道 B: 长度通道 (基于眉心到鼻尖) - 易受 Pitch 影响，受 Yaw 影响小
        
        # 获取长度通道关键点
        # 1: Nose Tip, 168: Glabella (Between Eyes/Eyebrow Center)
        # 注意：用户指定眉心到鼻尖距离为 8cm (FACE_REF_LENGTH_CM)
        p_nose_tip = self._extract_landmark_point(face_landmarks, 1, w, h)
        p_brow_center = self._extract_landmark_point(face_landmarks, 168, w, h)

        # 滤波关键点 (使用 Keypoint 默认参数)
        f_nose_tip = None
        f_brow_center = None
        
        if p_nose_tip and p_brow_center:
             f_nose_tip = (self._get_filter('nose_tip_x', p_nose_tip[0], current_time), 
                           self._get_filter('nose_tip_y', p_nose_tip[1], current_time))
             f_brow_center = (self._get_filter('brow_center_x', p_brow_center[0], current_time), 
                              self._get_filter('brow_center_y', p_brow_center[1], current_time))

        # 计算像素距离
        # 宽度 (Outer Eyes)
        d_width_px = math.sqrt((f_outer_l[0] - f_outer_r[0])**2 + (f_outer_l[1] - f_outer_r[1])**2)
        
        # 长度 (Brow to Nose)
        d_length_px = 0
        if f_nose_tip and f_brow_center:
            d_length_px = math.sqrt((f_nose_tip[0] - f_brow_center[0])**2 + (f_nose_tip[1] - f_brow_center[1])**2)

        # Focal length calculation (优化：计算一次，传递给 update_offset)
        fov_rad = math.radians(camera_fov)
        focal_length = (w / 2.0) / math.tan(fov_rad / 2.0)
        
        # 3. Calculate Head Pose (PnP) First (needed for fusion weights)
        pitch, yaw, roll, rvec, tvec, rmat = self._calculate_head_pose(face_landmarks, w, h, cam_matrix, dist_coeffs, current_time)
        
        # --- Filter Yaw & Pitch ---
        yaw_config = settings.FILTER_CONFIG['FACE']['YAW']
        yaw = self._get_filter(
            'head_yaw', yaw, current_time,
            min_cutoff=yaw_config['min_cutoff'], 
            beta=yaw_config['beta'],
            d_cutoff=yaw_config['d_cutoff']
        )
        
        pitch_config = settings.FILTER_CONFIG['FACE'].get('PITCH', yaw_config) # Fallback to yaw config if not found
        pitch = self._get_filter(
            'head_pitch', pitch, current_time,
            min_cutoff=pitch_config['min_cutoff'], 
            beta=pitch_config['beta'],
            d_cutoff=pitch_config['d_cutoff']
        )

        # 4. Dual Channel Depth Estimation & Fusion
        # --------------------------------------------------------------------------------
        
        # 投影修正系数 (Projection Correction)
        # 当发生旋转时，2D 投影长度缩短，导致深度估算偏大
        # Width Channel affects by Yaw
        # Length Channel affects by Pitch
        
        # 防止除零
        cos_yaw = abs(math.cos(math.radians(yaw)))
        cos_pitch = abs(math.cos(math.radians(pitch)))
        
        if cos_yaw < 0.2: cos_yaw = 0.2
        if cos_pitch < 0.2: cos_pitch = 0.2

        # 初步估算各通道深度 (Assuming frontal, then correcting projection)
        # Z = (f * Real) / (Pixel / cos(angle)) = (f * Real * cos(angle)) / Pixel
        
        z_width = 0
        if d_width_px > 0:
            z_width = (focal_length * self.calibrated_width_cm * cos_yaw) / d_width_px

        z_length = 0
        if d_length_px > 0:
            z_length = (focal_length * settings.FACE_REF_LENGTH_CM * cos_pitch) / d_length_px

        # --- 动态校准 (Dynamic Calibration) ---
        # 策略：以长度通道 (8cm) 为基准，校准宽度通道的物理宽度
        # 条件：姿态端正 (Low Yaw/Pitch)
        is_stable_pose = (abs(yaw) < self.calibration_config['min_valid_yaw']) and \
                         (abs(pitch) < self.calibration_config['min_valid_pitch'])
        
        if is_stable_pose and z_length > 0 and d_width_px > 0:
            # 反推当前的"真实"宽度
            # Real_Width = (Z_length * Pixel_Width) / (f * cos_yaw)
            # 在 stable pose 下，cos_yaw ~ 1
            current_estimated_width = (z_length * d_width_px) / (focal_length * cos_yaw)
            
            # EMA 更新
            alpha = self.calibration_config['width_correction_alpha']
            self.calibrated_width_cm = (1 - alpha) * self.calibrated_width_cm + alpha * current_estimated_width

        # --- 深度融合 (Fusion) ---
        # 权重取决于角度：角度越大，该通道投影变形越大，权重应越低。
        # 当面部正对镜头时 (Yaw=0, Pitch=0), cos=1, 权重相等 (各占 0.5)。
        
        w_width = cos_yaw
        w_length = cos_pitch
        
        # 如果某个通道无效
        if z_width <= 0: w_width = 0
        if z_length <= 0: w_length = 0
        
        estimated_dist = 0
        total_weight = w_width + w_length
        
        if total_weight > 0:
            # 归一化权重，确保和为 1
            w_width_norm = w_width / total_weight
            w_length_norm = w_length / total_weight
            estimated_dist = w_width_norm * z_width + w_length_norm * z_length
        else:
            estimated_dist = self.current_estimated_dist # Fallback
            w_width_norm = 0
            w_length_norm = 0
            
        # 记录详细信息用于可视化调试
        self.current_depth_details = {
            'z_width': z_width,
            'z_length': z_length,
            'w_width': w_width_norm,
            'w_length': w_length_norm,
            'calibrated_width': self.calibrated_width_cm
        }
            
        # --- Z 值滤波 (Level 3) ---
        # 在计算 XY 偏移之前，先对 Z 进行滤波，防止 Z 的抖动引起 XY 的抖动
        z_config = settings.FILTER_CONFIG['FACE'].get('Z_VAL', settings.FILTER_CONFIG['FACE']['DISTANCE'])
        estimated_dist = self._get_filter(
            'face_z_val', estimated_dist, current_time,
            min_cutoff=z_config['min_cutoff'],
            beta=z_config['beta'],
            d_cutoff=z_config['d_cutoff']
        )
        
        # 兼容旧接口 (pixel_dist 仅用于显示或调试，这里取加权平均的等效像素值)
        # Pixel = f * Real / Z
        pixel_dist = 0
        if estimated_dist > 0:
            pixel_dist = (focal_length * self.calibrated_width_cm) / estimated_dist

        # Apply secondary smoothing (Optional, keeping for legacy compatibility if needed, but Z is already filtered)
        # self.current_pixel_dist = pixel_dist
        self.current_estimated_dist = estimated_dist
        self.current_pixel_dist = pixel_dist # Update for display

        # 5. Update offset
        # 优化：传递已计算的 focal_length，避免重复计算
        # 计算头部中心位置 (双眼中心/鼻梁)，而非使用右眼
        # MediaPipe Landmark 168: Point between eyes
        center_168 = self._extract_landmark_point(face_landmarks, 168, w, h)
        
        if center_168:
            # 应用 OneEuroFilter 滤波
            f_center_x = self._get_filter('head_center_x', center_168[0], current_time)
            f_center_y = self._get_filter('head_center_y', center_168[1], current_time)
            tracking_point = (f_center_x, f_center_y)
        else:
            # 回退方案：如果没有 168，使用之前计算的眼部中点
            stable_rx = (f_inner_r[0] + f_outer_r[0]) / 2.0
            stable_ry = (f_inner_r[1] + f_outer_r[1]) / 2.0
            tracking_point = (stable_rx, stable_ry)

        self.update_offset(tracking_point, w, h, pixel_dist, estimated_dist, focal_length=focal_length)
        
        # 记录 Yaw/Pitch 以便可视化
        self.current_yaw = yaw
        self.current_pitch = pitch
        
        return {
            'eye_points': eye_points,
            'raw_eye_points': raw_eye_points,
            'rvec': rvec,
            'tvec': tvec,
            'yaw': yaw,
            'pitch': pitch,
            'roll': roll,
            'rmat': rmat,
            'calibrated_width': self.calibrated_width_cm # Return for debug/display
        }

    def _calculate_head_pose(self, face_landmarks, w, h, cam_matrix, dist_coeffs, current_time):
        """Calculates head pose using PnP with simplified 4-point model."""
        # 4-Point Model (defined in settings.py):
        # 1: Nose Tip (Index 1)
        # 2: Brow Center (Index 168)
        # 3: Left Eye Outer (Index 33)
        # 4: Right Eye Outer (Index 263)
        
        indices = [1, 168, 33, 263]
        
        # 优化：预分配 numpy 数组
        image_points = np.empty((len(indices), 2), dtype=np.float64)
        
        for i, idx in enumerate(indices):
            pt = self._extract_landmark_point(face_landmarks, idx, w, h)
            if pt is None:
                return 0, 0, 0, None, None, None
            
            # --- 应用 OneEuroFilter ---
            filtered_x = self._get_filter(
                f'lm_{idx}_x', pt[0], current_time
            )
            filtered_y = self._get_filter(
                f'lm_{idx}_y', pt[1], current_time
            )
            
            image_points[i] = [filtered_x, filtered_y]
            
        # PnP 求解
        # 使用 SOLVEPNP_EPNP 或 SOLVEPNP_ITERATIVE
        # 注意：EPnP 需要至少 4 个点。DLT 需要至少 6 个点。
        # 当点数较少时，Iterative 或 P3P (如果是 3 或 4 点) 是更好的选择。
        # OpenCV 文档指出，对于 4 个点，SOLVEPNP_EPNP 或 SOLVEPNP_ITERATIVE 应该工作。
        # 但是，如果 SOLVEPNP_ITERATIVE 内部回退到 DLT，就会报错。
        # SOLVEPNP_P3P (flags=cv2.SOLVEPNP_P3P) 专门用于 4 个点 (3+1)。
        
        flags = cv2.SOLVEPNP_ITERATIVE
        if len(settings.MODEL_POINTS) == 4:
             flags = cv2.SOLVEPNP_P3P # P3P requires exactly 4 points
             # 注意：P3P 返回多个解，solvePnP 可能只返回其中一个。
             # 实际上，SOLVEPNP_EPNP 对于 n >= 4 也是稳定的。
             flags = cv2.SOLVEPNP_EPNP
             
        # 强制使用 EPnP，它对 4 个点也有效且比 DLT 稳定
        (success, rotation_vector, translation_vector) = cv2.solvePnP(
            settings.MODEL_POINTS, 
            image_points, 
            cam_matrix, 
            dist_coeffs, 
            flags=cv2.SOLVEPNP_EPNP
        )
    
        if not success:
            return 0, 0, 0, None, None, None
        
        # 计算欧拉角
        rmat, jac = cv2.Rodrigues(rotation_vector)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
        
        return angles[0], angles[1], angles[2], rotation_vector, translation_vector, rmat

    def calculate_single_eye_gaze(self, iris_center_2d, eye_center_model_3d, rvec, tvec, cam_matrix, dist_coeffs, eye_radius=60.0, rmat=None):
        """
        计算单眼视线向量
        :param iris_center_2d: (x, y) 像素坐标
        :param eye_center_model_3d: (x, y, z) 模型坐标系下的眼球中心
        :param rvec: 头部旋转向量
        :param tvec: 头部平移向量
        :param cam_matrix: 相机内参
        :param rmat: 可选，预计算的旋转矩阵
        :return: (gaze_vector_3d, eye_center_cam_3d) 相机坐标系下的视线向量和眼球中心
        """
        # 1. 将眼球中心变换到相机坐标系
        # 优化：复用 rmat
        if rmat is None:
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


    def update_offset(self, tracking_point, frame_width, frame_height, pixel_dist, real_dist_cm, fov=60.0, focal_length=None):
        """
        计算并更新头部中心（画面中双眼中心）相对于摄像机光轴的物理偏移
        采用针孔成像模型 (Pinhole Camera Model):
        X = Z * (x - cx) / fx
        Y = Z * (y - cy) / fy
        
        tracking_point: (u, v) 稳定的头部中心点（MediaPipe 168）
        """
        # 如果距离无效，尝试使用缓存
        if real_dist_cm <= 0:
            if self.current_estimated_dist > 0:
                real_dist_cm = self.current_estimated_dist
            else:
                return

        if tracking_point is None:
            return
            
        # 1. 确定头部中心坐标 (u, v)
        # 使用传入的稳定跟踪点
        u, v = tracking_point
        self.head_center_pos = (int(u), int(v))
        
        # 2. 确定相机内参 (Intrinsics)
        # 主点 (Principal Point) 坐标 (cx, cy)
        # 严格定义为图像中心，消除系统性偏差
        cx = frame_width / 2.0
        cy = frame_height / 2.0
        
        # 焦距 (Focal Length) fx, fy
        # 优化：优先使用传入的 focal_length
        if focal_length is not None:
            fx = focal_length
            fy = focal_length
        else:
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

    def get_gaze_params(self):
        """批量获取视线参数，减少属性访问开销"""
        return self.current_estimated_dist, self.current_offset_x, self.current_offset_y
