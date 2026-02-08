import cv2
import numpy as np
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import socket
import threading
import queue
# from scipy.interpolate import CubicSpline

# 全局变量用于线程间通信
VISUALIZE = True # 设为 False 时，跳过所有可视化绘制，只保留计算和 UDP 发送
latest_data = None
data_lock = threading.Lock()
stop_event = threading.Event()

# UDP socket (initialized in main)
sock = None

# --- Camera Model ---
class CameraModel:
    def __init__(self, frame_w, frame_h, fov_deg=60.0):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.fov = fov_deg
        
        # Calculate intrinsics once
        self.fov_rad = math.radians(self.fov)
        # Assuming square pixels and principal point at center
        self.focal_length = (self.frame_w / 2) / math.tan(self.fov_rad / 2)
        self.cx = self.frame_w / 2.0
        self.cy = self.frame_h / 2.0
        
        self.cam_matrix = np.array([
            [self.focal_length, 0, self.cx],
            [0, self.focal_length, self.cy],
            [0, 0, 1]
        ], dtype="double")
        self.dist_coeffs = np.zeros((4, 1))

# --- 1€ Filter Implementation for Stability & Responsiveness ---
class OneEuroFilter:
    def __init__(self, t0, x0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        """
        min_cutoff: 最小截止频率 (Hz)。越小越平滑，延迟越高。
        beta: 速度系数。越大则高速运动时响应越快（减少延迟），但低速时抖动可能增加。
        d_cutoff: 导数截止频率 (Hz)。通常设为 1.0 Hz。
        """
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = 0.0
        self.t_prev = float(t0)

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def filter(self, t, x):
        t_e = t - self.t_prev
        
        # 避免时间戳重复或乱序导致的问题
        if t_e <= 0.0:
            return self.x_prev

        # 计算导数（速度）并进行低通滤波
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        # 根据速度动态调整截止频率
        # 速度越快，cutoff 越高，过滤越少（减少延迟）
        # 速度越慢，cutoff 越低，过滤越多（减少抖动）
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        # 更新状态
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat

class OneDKalmanFilter:
    def __init__(self, Q=1e-5, R=0.01):
        self.kf = cv2.KalmanFilter(2, 1)
        self.kf.measurementMatrix = np.array([[1, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)
        # Process noise covariance (Q) - prediction uncertainty
        self.kf.processNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * Q
        # Measurement noise covariance (R) - measurement uncertainty
        self.kf.measurementNoiseCov = np.array([[1]], np.float32) * R
        self.kf.statePost = np.array([[0], [0]], np.float32)
        self.first_run = True

    def update(self, measurement):
        if self.first_run:
            self.kf.statePost = np.array([[measurement], [0]], np.float32)
            self.first_run = False
        
        self.kf.predict()
        self.kf.correct(np.array([[measurement]], np.float32))
        return self.kf.statePost[0][0]

class EyeTracker:
    def __init__(self):
        self.current_pixel_dist = 0
        self.current_estimated_dist = 0
        self.current_offset_x = 0
        self.current_offset_y = 0
        
        # Initialize Kalman filters for distance smoothing (Secondary Layer)
        self.pixel_dist_filter = OneDKalmanFilter(Q=0.1, R=5.0)
        self.real_dist_filter = OneDKalmanFilter(Q=0.1, R=5.0)
        # Filters for offset (Secondary Layer)
        self.offset_x_filter = OneDKalmanFilter(Q=0.1, R=0.1)
        self.offset_y_filter = OneDKalmanFilter(Q=0.1, R=0.1)
        
        # 记录主视眼位置用于绘制
        self.dominant_eye_pos = None

        # --- Pixel Level Filters (Primary Layer) ---
        # 使用 OneEuroFilter 在源头消除抖动
        # 参数调整建议：min_cutoff 越小越稳，beta 越大越快
        self.filters_initialized = False
        self.lx_filter = None
        self.ly_filter = None
        self.rx_filter = None
        self.ry_filter = None
    
    def reset(self):
        self.dominant_eye_pos = None
        self.filters_initialized = False
        # Keep current distance values until new ones are calculated to avoid flickering
    
    def filter_eye_points(self, eye_points):
        """
        对原始像素坐标进行滤波
        eye_points: [(lx, ly), (rx, ry)]
        """
        current_time = time.time()
        
        if len(eye_points) < 2:
            return eye_points
            
        lx, ly = eye_points[0]
        rx, ry = eye_points[1]

        if not self.filters_initialized:
            # min_cutoff=0.5: 静态时非常平滑
            # beta=0.1: 移动时有一定响应速度，避免过度延迟
            self.lx_filter = OneEuroFilter(current_time, lx, min_cutoff=0.5, beta=0.1)
            self.ly_filter = OneEuroFilter(current_time, ly, min_cutoff=0.5, beta=0.1)
            self.rx_filter = OneEuroFilter(current_time, rx, min_cutoff=0.5, beta=0.1)
            self.ry_filter = OneEuroFilter(current_time, ry, min_cutoff=0.5, beta=0.1)
            self.filters_initialized = True
            return eye_points
        
        # 应用滤波
        f_lx = self.lx_filter.filter(current_time, lx)
        f_ly = self.ly_filter.filter(current_time, ly)
        f_rx = self.rx_filter.filter(current_time, rx)
        f_ry = self.ry_filter.filter(current_time, ry)
        
        return [(f_lx, f_ly), (f_rx, f_ry)]

    def apply_distance_filter(self, pixel_dist, estimated_dist):
        """应用Kalman滤波处理距离数据"""
        # 数据有效性检查
        if pixel_dist <= 0 or estimated_dist <= 0:
            return self.current_pixel_dist, self.current_estimated_dist
            
        filtered_pixel = self.pixel_dist_filter.update(pixel_dist)
        filtered_estimated = self.real_dist_filter.update(estimated_dist)
        
        return filtered_pixel, filtered_estimated

    def update_offset(self, eye_points, frame_width, frame_height, pixel_dist, real_dist_cm):
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
        fov = 60.0
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
        
        # 调试打印
        scale_factor = Z / fx if fx != 0 else 0
        # print(f"px_offset_x: {u - cx:.2f}, px_offset_y: {v - cy:.2f}, Scale: {scale_factor:.4f} cm/px")
        
        # 滤波 (Keep Secondary Filter)
        filtered_x = self.offset_x_filter.update(real_offset_x)
        filtered_y = self.offset_y_filter.update(real_offset_y)
        
        # 保留浮点数精度
        self.current_offset_x = filtered_x
        self.current_offset_y = filtered_y

class ImagePreprocessor:
    def __init__(self):
        self.last_roi = None # (x, y, w, h)
        self.alpha = 0.7 # ROI smoothing factor (0-1)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def process(self, frame, last_landmarks=None):
        """
        Preprocessing: ROI Crop -> Upscale -> Bilateral Filter -> Contrast Enhancement
        Returns: processed_frame, roi_info (x, y, w, h, scale)
        """
        h_frame, w_frame = frame.shape[:2]
        
        # 1. Compute ROI
        roi = self._compute_roi(w_frame, h_frame, last_landmarks)
        x, y, w, h = roi
        
        # Crop
        if w <= 0 or h <= 0:
            return frame, (0, 0, w_frame, h_frame, 1.0)
            
        crop = frame[y:y+h, x:x+w]
        
        if crop.size == 0:
            return frame, (0, 0, w_frame, h_frame, 1.0)

        # 2. Pixel Alignment / Cubic Interpolation Upscaling
        scale_factor = 1.0
        target_min_size = 256
        min_dim = min(w, h)
        
        if min_dim < target_min_size:
            scale_factor = target_min_size / min_dim
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 3. Bilateral Filter (Edge-preserving smoothing)
        # crop = cv2.bilateralFilter(crop, d=5, sigmaColor=75, sigmaSpace=75)
        
        # 4. Contrast Enhancement (CLAHE on L channel)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l2 = self.clahe.apply(l)
        lab = cv2.merge((l2, a, b))
        crop = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return crop, (x, y, w, h, scale_factor)

    def _compute_roi(self, w_frame, h_frame, landmarks):
        if landmarks is None:
            # Lost target or initial state: Smoothly reset to full frame
            target_roi = (0, 0, w_frame, h_frame)
        else:
            # Compute bounding box
            xs = [p.x for p in landmarks]
            ys = [p.y for p in landmarks]
            
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            
            # Convert to pixel coordinates (center and dimensions)
            cx = (min_x + max_x) / 2 * w_frame
            cy = (min_y + max_y) / 2 * h_frame
            fw = (max_x - min_x) * w_frame
            fh = (max_y - min_y) * h_frame
            
            # Expand range (Padding)
            padding = 2.0 
            size = max(fw, fh) * padding
            
            # Calculate top-left
            x = int(cx - size / 2)
            y = int(cy - size / 2)
            w = int(size)
            h = int(size)
            
            # Boundary checks
            x = max(0, x)
            y = max(0, y)
            w = min(w, w_frame - x)
            h = min(h, h_frame - y)
            
            target_roi = (x, y, w, h)

        # Smooth ROI changes
        if self.last_roi is None:
            self.last_roi = target_roi
            return target_roi
        
        lx, ly, lw, lh = self.last_roi
        tx, ty, tw, th = target_roi
        
        # Faster reset if target lost
        alpha = self.alpha if landmarks is not None else 0.5
        
        nx = int(lx * alpha + tx * (1 - alpha))
        ny = int(ly * alpha + ty * (1 - alpha))
        nw = int(lw * alpha + tw * (1 - alpha))
        nh = int(lh * alpha + th * (1 - alpha))
        
        # Boundary safety check
        nx = max(0, nx)
        ny = max(0, ny)
        nw = min(nw, w_frame - nx)
        nh = min(nh, h_frame - ny)
        
        self.last_roi = (nx, ny, nw, nh)
        return self.last_roi

    def restore_landmarks(self, detection_result, roi_info, w_frame, h_frame):
        """Restore normalized local coordinates to normalized global coordinates"""
        if not detection_result.face_landmarks:
            return

        roi_x, roi_y, roi_w, roi_h, scale = roi_info
        
        # If full frame and no scale, no processing needed
        if roi_x == 0 and roi_y == 0 and roi_w == w_frame and roi_h == h_frame and scale == 1.0:
            return

        for face_landmarks in detection_result.face_landmarks:
            for p in face_landmarks:
                # Convert back to global normalized coordinates
                # p.x is normalized in the cropped image
                # roi_w is the width of the crop in the original image
                # roi_x is the x offset in the original image
                
                new_x = (p.x * roi_w + roi_x) / w_frame
                new_y = (p.y * roi_h + roi_y) / h_frame
                
                p.x = new_x
                p.y = new_y

# 计算头部姿态（Yaw角）用于距离校正
def get_head_pose(face_landmarks, w, h, cam_matrix, dist_coeffs):
    # 2D 图像点 (使用 MediaPipe 关键点索引)
    # 扩展至 11 个关键点以提高稳定性
    # 1: Nose Tip
    # 152: Chin
    # 33: Left Eye Outer
    # 263: Right Eye Outer
    # 61: Left Mouth Corner
    # 291: Right Mouth Corner
    # 133: Left Eye Inner
    # 362: Right Eye Inner
    # 70: Left Eyebrow Outer
    # 300: Right Eyebrow Outer
    # 2: Nose Bottom
    
    image_points = np.array([
        (face_landmarks[1].x * w, face_landmarks[1].y * h),     # Nose tip
        (face_landmarks[152].x * w, face_landmarks[152].y * h), # Chin
        (face_landmarks[33].x * w, face_landmarks[33].y * h),   # Left eye outer
        (face_landmarks[263].x * w, face_landmarks[263].y * h), # Right eye outer
        (face_landmarks[61].x * w, face_landmarks[61].y * h),   # Left mouth corner
        (face_landmarks[291].x * w, face_landmarks[291].y * h), # Right mouth corner
        (face_landmarks[133].x * w, face_landmarks[133].y * h), # Left eye inner
        (face_landmarks[362].x * w, face_landmarks[362].y * h), # Right eye inner
        (face_landmarks[70].x * w, face_landmarks[70].y * h),   # Left eyebrow outer
        (face_landmarks[300].x * w, face_landmarks[300].y * h), # Right eyebrow outer
        (face_landmarks[2].x * w, face_landmarks[2].y * h)      # Nose bottom
    ], dtype="double")

    # 3D 模型点 (通用人脸模型，单位：任意单位，比例约 50 units/cm)
    # 坐标系：X 右, Y 上, Z 前 (相对面部中心)
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -500.0, -300.0),       # Chin
        (-225.0, 170.0, -135.0),     # Left eye outer
        (225.0, 170.0, -135.0),      # Right eye outer
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0),     # Right mouth corner
        (-90.0, 170.0, -120.0),      # Left eye inner
        (90.0, 170.0, -120.0),       # Right eye inner
        (-250.0, 300.0, -100.0),     # Left eyebrow outer
        (250.0, 300.0, -100.0),      # Right eyebrow outer
        (0.0, -80.0, -50.0)          # Nose bottom
    ], dtype="double")

    # PnP 求解
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)

    if not success:
        return 0, 0, 0, None, None

    # 计算欧拉角
    rmat, jac = cv2.Rodrigues(rotation_vector)
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
    
    # angles[0]=pitch, angles[1]=yaw, angles[2]=roll
    return angles[0], angles[1], angles[2], rotation_vector, translation_vector

def calculate_single_eye_gaze(iris_center_2d, eye_center_model_3d, rvec, tvec, cam_matrix, dist_coeffs):
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
    # Undistort points first? Assuming small distortion or raw coordinates for simplicity
    # Ray direction: inv(K) * [u, v, 1]
    p_iris_h = np.array([iris_center_2d[0], iris_center_2d[1], 1.0])
    ray_dir = np.dot(np.linalg.inv(cam_matrix), p_iris_h)
    ray_dir = ray_dir / np.linalg.norm(ray_dir) # Normalize
    
    # 3. 计算射线与眼球球面的交点 (或者近似)
    # Sphere: Center = eye_center_cam, Radius = 60 (12mm * 50 units/cm)
    radius = 60.0
    
    # Ray: O = (0,0,0), D = ray_dir
    # Intersection: |t*D - C|^2 = r^2
    # t^2 - 2(C.D)t + |C|^2 - r^2 = 0
    C = eye_center_cam
    a = 1.0
    b = -2.0 * np.dot(C, ray_dir)
    c_val = np.dot(C, C) - radius**2
    
    discriminant = b**2 - 4*a*c_val
    
    if discriminant >= 0:
        # Ray intersects sphere
        t1 = (-b - math.sqrt(discriminant)) / (2*a)
        t2 = (-b + math.sqrt(discriminant)) / (2*a)
        t = min(t1, t2) if t1 > 0 and t2 > 0 else max(t1, t2) # Should be positive
        
        if t > 0:
            iris_on_sphere = t * ray_dir
            gaze_vector = iris_on_sphere - eye_center_cam
            return gaze_vector, eye_center_cam
            
    # Fallback: 如果没有交点（计算误差），使用射线上的最近点
    # Closest point on ray to C is at t = C.D
    t_closest = np.dot(C, ray_dir)
    closest_point = t_closest * ray_dir
    # 强制长度为半径
    vec = closest_point - eye_center_cam
    vec_norm = np.linalg.norm(vec)
    if vec_norm > 0:
        gaze_vector = vec / vec_norm * radius
    else:
        # Fallback to head forward direction (Z-axis of head in camera space)
        # Head Z-axis in camera space is the 3rd column of R
        gaze_vector = rmat[:, 2] * radius
        
    return gaze_vector, eye_center_cam

def calculate_gaze_intersection(P_L, V_L, P_R, V_R):
    """
    计算左右眼视线的交点（或最接近点），以右眼视线为基准。
    即计算两条异面直线公垂线在右眼视线上的垂足。
    
    :param P_L: 左眼位置 (3,)
    :param V_L: 左眼视线向量 (3,) (normalized)
    :param P_R: 右眼位置 (3,)
    :param V_R: 右眼视线向量 (3,) (normalized)
    :return: intersection_point (3,), is_converging (bool)
    """
    # 确保向量归一化
    V_L = V_L / np.linalg.norm(V_L)
    V_R = V_R / np.linalg.norm(V_R)
    
    # P_L + t * V_L = P_R + s * V_R + (shortest_dist_vec)
    # 我们想求 s (on right ray) 使得距离最小
    
    w0 = P_L - P_R
    a = np.dot(V_L, V_L) # 1
    b = np.dot(V_L, V_R)
    c = np.dot(V_R, V_R) # 1
    d = np.dot(V_L, w0)
    e = np.dot(V_R, w0)
    
    denom = a * c - b * b
    
    if denom < 1e-6:
        # 平行，无交点。假设看无穷远
        # 或者设为一个默认距离，比如 1米
        # 即使平行，也返回一个点供显示
        return P_R + V_R * 1000.0, False

    # s = (b*d - a*e) / denom
    s = (b * d - a * e) / denom
    t = (b * e - c * d) / denom
    
    # 宽松的交点判断：
    # 不再强制要求 s > 0 (在前方) 才能算作交点，因为用户要求"无论是否有交点...持续显示"
    # 但我们仍然可以通过 is_converging 标记来区分是否合理
    
    # 计算公垂线中点作为交点
    P_on_L = P_L + t * V_L
    P_on_R = P_R + s * V_R
    intersection = (P_on_L + P_on_R) / 2.0
    
    # 判断是否"合理汇聚" (仅用于UI颜色显示)
    # 1. 交点在前方一定距离 (s > 50 units = 1cm)
    # 2. 且公垂线长度(两条射线最近距离)不能太大? (用户说无法完美相交，所以忽略距离)
    if s > 50.0 and t > 50.0:
        is_converging = True
    else:
        is_converging = False
        
    return intersection, is_converging
    closest_point = t_closest * ray_dir
    # 强制长度为半径
    vec = closest_point - eye_center_cam
    vec_norm = np.linalg.norm(vec)
    if vec_norm > 0:
        gaze_vector = vec / vec_norm * radius
    else:
        # Fallback to head forward direction (Z-axis of head in camera space)
        # Head Z-axis in camera space is the 3rd column of R
        gaze_vector = rmat[:, 2] * radius
        
    return gaze_vector, eye_center_cam

# 计算瞳孔间距和距离估算
def calculate_distance(eye_points, frame_width, frame_height, fov=55):
    if len(eye_points) < 2:
        return 0, 0
    
    # 计算瞳孔间距（像素）
    point1, point2 = eye_points[0], eye_points[1]
    pixel_distance = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    # 已知平均瞳距6.5cm
    real_pupil_distance = 7  # cm
    
    # 计算摄像头的焦距（基于视角和图像宽度）
    fov_rad = math.radians(fov)
    focal_length = (frame_width / 2) / math.tan(fov_rad / 2)
    
    # 估算距离
    if pixel_distance > 0:
        estimated_distance = (real_pupil_distance * focal_length) / pixel_distance
    else:
        estimated_distance = 0
    
    return pixel_distance, estimated_distance

# 初始化跟踪器
tracker = EyeTracker()
preprocessor = ImagePreprocessor()
last_landmarks_norm = None

# Initialize MediaPipe Face Landmarker
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=vision.RunningMode.VIDEO)
detector = vision.FaceLandmarker.create_from_options(options)

import json
import os

# --- 持久化存储管理 ---

class ConfigManager:
    def __init__(self, config_dir="config"):
        self.config_dir = config_dir
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)
            
        self.cameras_config_path = os.path.join(self.config_dir, "cameras.json")
        self.user_config_path = os.path.join(self.config_dir, "user_prefs.json")
        
        self.cameras_data = self._load_json(self.cameras_config_path)
        self.user_prefs = self._load_json(self.user_config_path)

    def _load_json(self, path):
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_json(self, path, data):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    # --- 摄像头数据管理 ---
    def update_camera(self, device_index, fov=None, name=None, user_configured=False, resolution=None, exposure=None):
        idx_str = str(device_index)
        if idx_str not in self.cameras_data:
            self.cameras_data[idx_str] = {
                "name": f"Camera {device_index}",
                "fov": 60.0,
                "user_configured": False
            }
        
        if fov is not None:
            self.cameras_data[idx_str]["fov"] = fov
        if name is not None:
            self.cameras_data[idx_str]["name"] = name
        if user_configured:
            self.cameras_data[idx_str]["user_configured"] = True
        if resolution is not None:
            self.cameras_data[idx_str]["resolution"] = resolution
        if exposure is not None:
            self.cameras_data[idx_str]["exposure"] = exposure
            
        self._save_json(self.cameras_config_path, self.cameras_data)

    def get_camera_info(self, device_index):
        idx_str = str(device_index)
        return self.cameras_data.get(idx_str)

    # --- 用户偏好管理 ---
    def set_last_camera(self, index):
        self.user_prefs['last_camera_index'] = index
        self._save_json(self.user_config_path, self.user_prefs)

    def get_last_camera(self):
        return self.user_prefs.get('last_camera_index')

# --- 视频流获取优化 (Producer) ---
class WebcamVideoStream:
    def __init__(self, src=0, width=1920, height=1080, api_preference=cv2.CAP_ANY, queue_size=2, exposure=-5.0):
        self.src = src
        self.width = width
        self.height = height
        self.api_preference = api_preference
        self.exposure = exposure
        
        # 初始化摄像头
        self.stream = cv2.VideoCapture(self.src, self.api_preference)
        
        # 优化配置
        # 1. 强制 MJPEG 压缩
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        # 2. 设置分辨率
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # 3. 设置 FPS
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        # 4. 减少 OpenCV 内部缓冲区
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # 5. 关闭自动曝光、白平衡、增益
        try:
            self.stream.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) # 0.25 通常对应 'Manual' 模式 (具体取决于后端)
            self.stream.set(cv2.CAP_PROP_EXPOSURE, self.exposure) 
            self.stream.set(cv2.CAP_PROP_AUTO_WB, 0)
            self.stream.set(cv2.CAP_PROP_WB_TEMPERATURE, 4600)
            self.stream.set(cv2.CAP_PROP_GAIN, 32) 
            print(f"尝试设置曝光为 {self.exposure} (手动模式)...")
        except Exception as e:
            print(f"设置摄像头手动参数失败: {e}")
        
        # 检查是否成功打开
        if not self.stream.isOpened():
            print("WebcamVideoStream: 无法打开摄像头")
            self.stopped = True
        else:
            self.stopped = False
            
        # 读取第一帧确认
        (self.grabbed, self.frame) = self.stream.read()
        if not self.grabbed:
            print("WebcamVideoStream: 无法读取第一帧")
            self.stopped = True

        # 使用 Queue 替代简单的变量 (Thread-safe)
        # maxsize 限制队列长度，防止堆积
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.frame_id = 0

    def start(self):
        if self.stopped:
            return self
        print("启动视频流读取线程...")
        self.t = threading.Thread(target=self.update, args=())
        self.t.daemon = True
        self.t.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            
            # 读取下一帧
            grabbed, frame = self.stream.read()
            
            if not grabbed:
                self.stopped = True
                return

            self.frame_id += 1
            
            # 尝试放入队列，如果满了则移除旧的（非阻塞尝试）
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            
            try:
                self.frame_queue.put((frame, self.frame_id), block=False)
            except queue.Full:
                pass # Should not happen if we just removed one, but safe guard

    def read(self):
        # 非阻塞获取最新帧
        try:
            return True, self.frame_queue.get_nowait()
        except queue.Empty:
            return False, (None, -1)
    
    def get(self, propId):
        return self.stream.get(propId)

    def stop(self):
        self.stopped = True
        if hasattr(self, 't'):
            self.t.join()
        self.stream.release()

# --- 图像处理与检测线程 (Consumer 1 & Producer 2) ---
class FrameProcessorThread:
    def __init__(self, input_queue, output_queue, preprocessor, detector, stop_event):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.preprocessor = preprocessor
        self.detector = detector
        self.stop_event = stop_event
        self.last_landmarks_norm = None
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True

    def start(self):
        self.thread.start()

    def run(self):
        while not self.stop_event.is_set():
            try:
                # 阻塞等待输入，超时10ms避免死锁
                frame_data = self.input_queue.get(timeout=0.01)
                frame, frame_id = frame_data
                
                # 开始处理
                h, w = frame.shape[:2]
                
                # 预处理：ROI -> 放大 -> 滤波 -> 增强
                processed_frame, roi_info = self.preprocessor.process(frame, self.last_landmarks_norm)
                
                # 转换处理后的帧为 RGB
                processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # MediaPipe 处理
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=processed_rgb)
                timestamp_ms = int(time.time() * 1000)
                detection_result = self.detector.detect_for_video(mp_image, timestamp_ms)
                
                # 坐标还原 (Local Normalized -> Global Normalized)
                self.preprocessor.restore_landmarks(detection_result, roi_info, w, h)
                
                # 更新上一帧 Landmarks (用于下一帧 ROI 计算)
                # 注意：这里是线程内部状态，因为下一帧处理依赖上一帧结果
                if detection_result.face_landmarks:
                    self.last_landmarks_norm = detection_result.face_landmarks[0]
                else:
                    self.last_landmarks_norm = None
                
                # 将结果放入输出队列
                if self.output_queue.full():
                    try:
                        self.output_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.output_queue.put({
                    'frame': frame,
                    'frame_id': frame_id,
                    'detection_result': detection_result,
                    'roi_info': roi_info,
                    'processed_frame': processed_frame # Optional, for debug
                })
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing Error: {e}")



# 初始化管理器
config_manager = ConfigManager()

# 摄像头选择逻辑
def select_camera_device():
    print("正在扫描摄像头设备...")
    available_indices = []
    # 扫描常用索引 0-4
    for i in range(5):
        # 尝试快速打开检测 (优先 DSHOW)
        temp_cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not temp_cap.isOpened():
            temp_cap = cv2.VideoCapture(i, cv2.CAP_ANY)
        
        if temp_cap.isOpened():
            available_indices.append(i)
            # 自动存入/更新配置文件
            config_manager.update_camera(i)
            temp_cap.release()
    
    if not available_indices:
        print("错误：未检测到摄像头设备。")
        return None, 60.0
        
    selected_index = None
    
    # 尝试自动加载上次使用的摄像头
    last_index = config_manager.get_last_camera()
    if last_index is not None and last_index in available_indices:
        print(f"检测到上次使用的摄像头 (索引 {last_index})，按 Enter 自动选择，或输入其他索引...")
    
    if len(available_indices) == 1:
        print(f"检测到单个摄像头 (索引 {available_indices[0]})，自动连接。")
        selected_index = available_indices[0]
    else:
        print("检测到多个摄像头设备：")
        for idx in available_indices:
            info = config_manager.get_camera_info(idx)
            fov_display = ""
            name_display = ""
            if info:
                fov_display = f" (FOV: {info['fov']}°)"
                name_display = f" [{info['name']}]"
                
            marker = " *" if idx == last_index else ""
            print(f" - 设备索引: {idx}{name_display}{fov_display}{marker}")
            
        while True:
            try:
                prompt = f"请输入要使用的摄像头索引 {available_indices}"
                if last_index is not None and last_index in available_indices:
                    prompt += f" [默认 {last_index}]"
                prompt += ": "
                
                selection = input(prompt).strip()
                
                if not selection and last_index is not None and last_index in available_indices:
                    selected_index = last_index
                    break
                
                idx = int(selection)
                if idx in available_indices:
                    selected_index = idx
                    break
                else:
                    print("输入的索引无效，请重新输入。")
            except ValueError:
                print("输入格式错误，请输入数字索引。")
    
    # 保存选择
    config_manager.set_last_camera(selected_index)
    
    # 获取FOV参数
    saved_info = config_manager.get_camera_info(selected_index)
    default_fov = saved_info['fov'] if saved_info else 60.0
    
    # 检查是否已有用户配置
    if saved_info and saved_info.get("user_configured", False):
        print(f"检测到已保存的配置参数：FOV {default_fov}° (跳过输入)")
        return selected_index, default_fov
    
    while True:
        try:
            fov_input = input(f"请输入摄像头FOV(视场角) [默认{default_fov}]: ").strip()
            if not fov_input:
                fov_val = default_fov
            else:
                fov_val = float(fov_input)
            
            if 0 < fov_val < 180:
                # 用户确认输入后，更新配置文件并标记为已配置
                config_manager.update_camera(selected_index, fov=fov_val, user_configured=True)
                print(f"已保存配置：索引 {selected_index}, FOV {fov_val}°")
                return selected_index, fov_val
            else:
                print("FOV必须在0到180之间。")
        except ValueError:
            print("输入格式错误，请输入数字。")
    
    return selected_index, fov_val

def select_resolution(cap, camera_index, config_manager):
    # 1. 检查是否有保存的分辨率配置
    saved_info = config_manager.get_camera_info(camera_index)
    if saved_info and "resolution" in saved_info:
        res = saved_info["resolution"]
        print(f"检测到已保存的分辨率配置: {res[0]}x{res[1]}")
        return res[0], res[1]

    # 2. 如果没有保存配置，扫描支持的分辨率
    print("正在扫描摄像头支持的分辨率...")
    target_resolutions = [
        (1920, 1080), # 1080p
        (1280, 720),  # 720p
        (800, 600),   # SVGA
        (640, 480)    # VGA
    ]
    
    available_resolutions = []
    
    # 保存当前的设置以便恢复（可选，这里主要是为了测试）
    current_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    current_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    for w, h in target_resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        
        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 只有当实际分辨率与请求的一致时才认为是完美支持
        if aw == w and ah == h:
            available_resolutions.append((w, h))
        else:
            # 如果不完全匹配，但之前没有添加过这个实际分辨率，也可以添加
            # 但为了用户体验，只列出精准匹配的，除非一个都匹配不上
            pass
            
    # 如果没有扫描到精准匹配的，就回退到默认
    if not available_resolutions:
        print("未能检测到标准分辨率，将使用默认分辨率。")
        return int(current_w), int(current_h)
        
    print("请选择要使用的分辨率：")
    for i, (w, h) in enumerate(available_resolutions):
        print(f" {i}: {w}x{h}")
        
    selected_res = None
    while True:
        try:
            choice = input(f"请输入分辨率编号 [0-{len(available_resolutions)-1}]: ").strip()
            if not choice:
                # 默认选第一个（通常是最高的）
                selected_res = available_resolutions[0]
                break
            
            idx = int(choice)
            if 0 <= idx < len(available_resolutions):
                selected_res = available_resolutions[idx]
                break
            else:
                print("无效的编号。")
        except ValueError:
            print("请输入数字。")
            
    # 保存选择
    print(f"已选择分辨率: {selected_res[0]}x{selected_res[1]}，并保存到配置。")
    config_manager.update_camera(camera_index, resolution=selected_res)
    
    return selected_res[0], selected_res[1]

camera_index, camera_fov = select_camera_device()
if camera_index is None:
    exit()

# 打开摄像头，尝试不同的API
cap_temp = None
used_api = cv2.CAP_ANY

for api in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
    cap_temp = cv2.VideoCapture(camera_index, api)
    if cap_temp.isOpened():
        print(f"检测到可用 API: {api}")
        used_api = api
        break

if not cap_temp or not cap_temp.isOpened():
    print(f"Error: Could not open camera {camera_index}")
    exit()

# 设置摄像头参数
# 使用用户选择或保存的分辨率
# 注意：select_resolution 会操作 cap_temp 来测试分辨率
target_w, target_h = select_resolution(cap_temp, camera_index, config_manager)

# 获取已保存的曝光配置，如果没有则使用默认值 -5.0
camera_info = config_manager.get_camera_info(camera_index)
exposure_val = -5.0
if camera_info and "exposure" in camera_info:
    exposure_val = float(camera_info["exposure"])
    print(f"检测到已保存的曝光配置: {exposure_val}")
else:
    # 也可以选择在这里让用户输入，或者只是使用默认值并保存
    # 为了方便，这里先只使用默认值，如果用户想改，可以直接改 JSON
    print(f"使用默认曝光值: {exposure_val}")
    # 第一次运行时可以保存一下默认值，方便用户后续在 json 中找到并修改
    config_manager.update_camera(camera_index, exposure=exposure_val)

# 释放临时 cap，准备使用 WebcamVideoStream 接管
cap_temp.release()

print(f"正在启动优化视频流 (MJPEG, 独立线程)...")
print(f"目标分辨率: {target_w}x{target_h}")

# 初始化多线程视频流
video_stream = WebcamVideoStream(src=camera_index, width=target_w, height=target_h, api_preference=used_api, exposure=exposure_val).start()

# 等待摄像头预热
time.sleep(1.0)

# 读取最终实际分辨率
actual_w = video_stream.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_h = video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"摄像头最终实际分辨率: {int(actual_w)}x{int(actual_h)}")

# 如果设置失败，给出警告
if int(actual_w) != target_w or int(actual_h) != target_h:
    print(f"警告: 实际分辨率 ({int(actual_w)}x{int(actual_h)}) 与请求分辨率 ({target_w}x{target_h}) 不一致。")

# Camera Model Initialization (Intrinsics Pre-calculation)
camera_model = CameraModel(actual_w, actual_h, camera_fov)
cam_matrix = camera_model.cam_matrix
dist_coeffs = camera_model.dist_coeffs

# MediaPipe Iris Indices
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

# UDP Configuration
UDP_IP = "127.0.0.1"
UDP_PORT = 8888

# Initialize UDP Socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# sock.setblocking(False) # Removed non-blocking mode to ensure data sending
print(f"UDP socket initialized. Target: {UDP_IP}:{UDP_PORT}")

# --- Pipeline Initialization ---
# Queue for processed results (Process -> Main)
# Main loop will consume this
processed_queue = queue.Queue(maxsize=2)

# Start Processing Thread
# video_stream.frame_queue is the input
processing_thread = FrameProcessorThread(
    video_stream.frame_queue, 
    processed_queue, 
    preprocessor, 
    detector, 
    stop_event
)
processing_thread.start()

print("Pipeline started: Capture -> Process -> Main Loop")

# FPS 计算相关
prev_frame_time = 0
new_frame_time = 0
last_processed_frame_id = -1

while True:
    # 从处理线程获取结果
    try:
        # Timeout 1ms to allow UI updates even if no new frame
        result_data = processed_queue.get(timeout=0.001)
    except queue.Empty:
        # Check if user pressed ESC in opencv window (needs waitKey to process events)
        if VISUALIZE:
            if cv2.waitKey(1) & 0xFF == 27:
                break
        continue
    
    frame = result_data['frame']
    current_frame_id = result_data['frame_id']
    detection_result = result_data['detection_result']
    roi_info = result_data['roi_info']
    
    h, w = frame.shape[:2]
    
    # 检查是否是重复帧 (虽然 Queue 保证了顺序，但为了安全)
    if current_frame_id == last_processed_frame_id:
        continue
        
    last_processed_frame_id = current_frame_id
    
    # 计算 FPS
    new_frame_time = time.time()
    fps = 0
    if prev_frame_time > 0:
        delta = new_frame_time - prev_frame_time
        if delta > 0:
            fps = 1.0 / delta
    prev_frame_time = new_frame_time
    
    # 可视化 ROI (蓝色矩形)
    rx, ry, rw, rh, _ = roi_info
    if VISUALIZE and (rw < w or rh < h):
        cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (255, 0, 0), 1)
    
    eye_points = []
    
    if detection_result.face_landmarks:
        for face_landmarks in detection_result.face_landmarks:
            # 获取左右虹膜中心
            # Mesh point 468 is Left Iris Center (User's Left)
            # Mesh point 473 is Right Iris Center (User's Right)
            
            pt_left_iris = face_landmarks[468]
            pt_right_iris = face_landmarks[473]
            
            cx_left = int(pt_left_iris.x * w)
            cy_left = int(pt_left_iris.y * h)
            
            cx_right = int(pt_right_iris.x * w)
            cy_right = int(pt_right_iris.y * h)
            
            # 原始坐标
            raw_eye_points = [(cx_left, cy_left), (cx_right, cy_right)]
            
            # --- 核心修改：在像素层级应用 OneEuro 滤波 ---
            eye_points = tracker.filter_eye_points(raw_eye_points)
            
            if VISUALIZE:
                # 绘制虹膜中心 (使用滤波后的坐标绘制，以反馈真实追踪位置)
                f_p1, f_p2 = eye_points
                cv2.circle(frame, (int(f_p1[0]), int(f_p1[1])), 3, (0, 255, 0), -1, cv2.LINE_AA) # Green for filtered
                cv2.circle(frame, (int(f_p2[0]), int(f_p2[1])), 3, (0, 255, 0), -1, cv2.LINE_AA)
                
                # 绘制原始点作为对比 (红色)
                cv2.circle(frame, (cx_left, cy_left), 2, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (cx_right, cy_right), 2, (0, 0, 255), -1, cv2.LINE_AA)
            
            # 计算距离和偏移 (使用滤波后的坐标)
            pixel_dist, estimated_dist = calculate_distance(eye_points, w, h, fov=camera_fov)
            
            # 计算头部姿态并校正距离
            # 注意：get_head_pose 使用的是 camera_model 计算出的矩阵
            pitch, yaw, roll, rvec, tvec = get_head_pose(face_landmarks, w, h, cam_matrix, dist_coeffs)
            
            if VISUALIZE and rvec is not None and tvec is not None and current_frame_id % 6 == 0:
                # --- 新增：眼球视线向量计算 ---
                # 定义模型空间中的眼球中心 (Left: 33/133, Right: 263/362)
                # Z轴向头骨内部偏移 12mm (60 units)
                # Midpoints calculated previously: Left(-157.5, 170, -127.5), Right(157.5, 170, -127.5)
                # Apply Z offset: -127.5 - 60 = -187.5
                left_eye_center_model = np.array([-157.5, 170.0, -187.5])
                right_eye_center_model = np.array([157.5, 170.0, -187.5])
                
                # 获取虹膜 2D 坐标 (已在 eye_points 中, eye_points = [(cx_left, cy_left), (cx_right, cy_right)])
                # 注意：eye_points 是滤波后的，比较稳定，使用它们
                left_iris_2d = eye_points[0]
                right_iris_2d = eye_points[1]
                
                # 计算左右眼视线
                l_gaze_vec, l_eye_center_cam = calculate_single_eye_gaze(
                    left_iris_2d, left_eye_center_model, rvec, tvec, cam_matrix, dist_coeffs
                )
                r_gaze_vec, r_eye_center_cam = calculate_single_eye_gaze(
                    right_iris_2d, right_eye_center_model, rvec, tvec, cam_matrix, dist_coeffs
                )
                
                # --- 计算视线交点并校正左眼 ---
                intersection_point, is_converging = calculate_gaze_intersection(
                    l_eye_center_cam, l_gaze_vec,
                    r_eye_center_cam, r_gaze_vec
                )
                
                # 标准化左眼视线向量：使其指向交点（或最接近点）
                # 用户要求：取消左眼视线方向的标准化，将两条射线的最近交点视为交点。
                # Left Gaze = Intersection - Left Eye (REMOVED as per user request)
                # new_l_gaze = intersection_point - l_eye_center_cam
                # new_l_gaze_norm = np.linalg.norm(new_l_gaze)
                # if new_l_gaze_norm > 0:
                #    l_gaze_vec = (new_l_gaze / new_l_gaze_norm) * 60.0

                # --- 绘制视线 ---
                axis_length = 500.0 # Scale for visualization
                
                # 投影左眼 (使用校正后的向量)
                l_start_3d = l_eye_center_cam
                l_end_3d = l_eye_center_cam + l_gaze_vec * (axis_length / 60.0) # Normalize by radius (60)
                
                # 投影右眼 (保持原样)
                r_start_3d = r_eye_center_cam
                r_end_3d = r_eye_center_cam + r_gaze_vec * (axis_length / 60.0)
                
                # 投影交点
                # Intersection point is in Camera Space.
                
                points_to_project = np.array([l_start_3d, l_end_3d, r_start_3d, r_end_3d, intersection_point])
                projected_points, _ = cv2.projectPoints(points_to_project, np.zeros((3,1)), np.zeros((3,1)), cam_matrix, dist_coeffs)
                
                p_l_start = (int(projected_points[0][0][0]), int(projected_points[0][0][1]))
                p_l_end = (int(projected_points[1][0][0]), int(projected_points[1][0][1]))
                p_r_start = (int(projected_points[2][0][0]), int(projected_points[2][0][1]))
                p_r_end = (int(projected_points[3][0][0]), int(projected_points[3][0][1]))
                p_intersect = (int(projected_points[4][0][0]), int(projected_points[4][0][1]))
                
                cv2.line(frame, p_l_start, p_l_end, (255, 0, 0), 2)
                cv2.line(frame, p_r_start, p_r_end, (255, 0, 0), 2)
                
                # 绘制眼球中心 (黄色小点)
                cv2.circle(frame, p_l_start, 3, (0, 255, 255), -1)
                cv2.circle(frame, p_r_start, 3, (0, 255, 255), -1)
                
                # 绘制交点 (紫色)
                # if is_converging:
                #    cv2.circle(frame, p_intersect, 5, (255, 0, 255), -1)
                
                # 恒显示交点坐标文本
                # Camera Space: X (Right), Y (Down), Z (Forward)
                # User asked for X (Horizontal offset), Y (Vertical offset) relative to optical axis
                # Optical axis is Z-axis (X=0, Y=0). So we just display X and Y of intersection.
                # Convert units: 50 units = 1cm. 
                ix_cm = intersection_point[0] / 50.0
                iy_cm = intersection_point[1] / 50.0
                iz_cm = intersection_point[2] / 50.0
                
                inter_text = f"Gaze Target: X:{ix_cm:.1f} Y:{iy_cm:.1f} Z:{iz_cm:.1f} cm"
                text_color = (255, 0, 255) if is_converging else (0, 0, 255) # Purple if good, Red if diverging
                cv2.putText(frame, inter_text, (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)


            # 三角变换校正
            correction_factor = math.cos(math.radians(yaw))
            if correction_factor < 0.2: correction_factor = 0.2
            
            corrected_dist = estimated_dist * correction_factor
            
            # 应用距离滤波 (二级平滑，可保留或适当调小Q/R)
            filtered_pixel, filtered_estimated = tracker.apply_distance_filter(pixel_dist, corrected_dist)
            tracker.current_pixel_dist = filtered_pixel
            tracker.current_estimated_dist = filtered_estimated
            
            # 更新偏移量 (使用滤波后的坐标和距离)
            tracker.update_offset(eye_points, w, h, filtered_pixel, filtered_estimated)

            # Direct UDP Send
            try:
                # 格式: "distance,offset_x,offset_y"
                # 匹配 Unity 端 DataManager.cs 的 ParseAndSetData 解析逻辑 (CSV格式)
                data_str = f"{tracker.current_estimated_dist:.2f},{tracker.current_offset_x:.2f},{tracker.current_offset_y:.2f}"
                sock.sendto(data_str.encode('utf-8'), (UDP_IP, UDP_PORT))
            except Exception as e:
                print(f"UDP Send Error: {e}")
    else:
        # 如果未检测到人脸，重置滤波器状态以便下次快速收敛
        tracker.reset()

    if VISUALIZE:
        # --- 绘制信息 ---
        
        # 绘制光轴中心（十字准星）
        center_x, center_y = w // 2, h // 2
        cv2.line(frame, (center_x - 10, center_y), (center_x + 10, center_y), (0, 0, 255), 1)
        cv2.line(frame, (center_x, center_y - 10), (center_x, center_y + 10), (0, 0, 255), 1)

        # 绘制 FPS (左上角第一行)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 绘制主视眼标记（黄色圆圈 + 'R'）
        if tracker.dominant_eye_pos is not None and detection_result.face_landmarks:
            dx, dy = tracker.dominant_eye_pos
            if 0 <= dx < w and 0 <= dy < h:
                cv2.circle(frame, (dx, dy), 8, (0, 255, 255), 2)
                cv2.putText(frame, "R", (dx + 10, dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if tracker.current_pixel_dist > 0:
            if tracker.current_estimated_dist > 200:
                info_text = "too far!"
                cv2.putText(frame, info_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                info_text = f"Dist: {int(tracker.current_estimated_dist)}cm | PD: {int(tracker.current_pixel_dist)}px"
                offset_text = f"Offset X: {int(tracker.current_offset_x):+d}cm | Y: {int(tracker.current_offset_y):+d}cm"
                cv2.putText(frame, info_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, offset_text, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # 绘制瞳孔间距线 (使用滤波后的点)
                if len(eye_points) >= 2:
                    p1 = (int(eye_points[0][0]), int(eye_points[0][1]))
                    p2 = (int(eye_points[1][0]), int(eye_points[1][1]))
                    cv2.line(frame, p1, p2, (255, 255, 0), 2)

        # 显示结果
        cv2.imshow('Face and Eye Detection (MediaPipe)', frame)
        
        # 按ESC键退出
        if cv2.waitKey(1) & 0xFF == 27:
            break

# 释放资源
stop_event.set()
video_stream.stop()
cv2.destroyAllWindows()
