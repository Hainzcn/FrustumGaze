import cv2
import numpy as np
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import socket
import threading
from collections import deque
# from scipy.interpolate import CubicSpline

# Global variables for thread communication
data_lock = threading.Lock()
# Buffer to store (timestamp, dist, offset_x, offset_y)
# Storing enough history for interpolation
data_buffer = deque(maxlen=20)
is_running = True

# UDP Configuration
UDP_IP = "127.0.0.1"
UDP_PORT = 8888
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def udp_sender_thread():
    """
    Thread to send upsampled data via UDP at 60Hz using Cubic Spline Interpolation
    """
    global is_running
    
    # Target frequency 60Hz
    interval = 1.0 / 60.0
    # Interpolation delay (seconds) to ensure we interpolate instead of extrapolate
    # Here 33.3ms at least, or it won't work
    delay = 0.04
    
    print("UDP Sender Thread Started (60Hz - Linear Interpolation)")
    
    while is_running:
        start_time = time.time()
        
        current_data = []
        with data_lock:
            # Need at least 2 points for linear interpolation
            if len(data_buffer) >= 2:
                current_data = list(data_buffer)
        
        if len(current_data) >= 2:
            # Target time for interpolation
            target_time = time.time() - delay
            
            # Find the two points bounding the target_time
            # data_buffer is ordered by time (oldest first)
            p1 = None
            p2 = None
            
            # Search for the interval [p1, p2] that contains target_time
            # Since current_data is sorted by time, we can iterate
            for i in range(len(current_data) - 1):
                if current_data[i][0] <= target_time <= current_data[i+1][0]:
                    p1 = current_data[i]
                    p2 = current_data[i+1]
                    break
            
            val_dist = 0
            val_x = 0
            val_y = 0
            
            found_interval = False
            if p1 is not None and p2 is not None:
                found_interval = True
                # Linear Interpolation (Lerp)
                t1, d1, x1, y1 = p1
                t2, d2, x2, y2 = p2
                
                # Calculate ratio (0.0 to 1.0)
                if t2 - t1 > 0.0001: # Avoid division by zero
                    ratio = (target_time - t1) / (t2 - t1)
                else:
                    ratio = 0
                
                val_dist = d1 + (d2 - d1) * ratio
                val_x = x1 + (x2 - x1) * ratio
                val_y = y1 + (y2 - y1) * ratio
            else:
                # Fallback if out of range
                if target_time < current_data[0][0]:
                    # Too old? Use oldest
                    _, val_dist, val_x, val_y = current_data[0]
                elif target_time > current_data[-1][0]:
                    # Too new? Use newest
                    _, val_dist, val_x, val_y = current_data[-1]
                else:
                    # Should be covered by loop, but just in case
                    _, val_dist, val_x, val_y = current_data[-1]

            try:
                # Send UDP
                # Format: distance,offset_x,offset_y (保留1位小数)
                message_str = f"{val_dist:.1f},{val_x:.1f},{val_y:.1f}"
                sock.sendto(message_str.encode('utf-8'), (UDP_IP, UDP_PORT))
                # if found_interval:
                #    print(f"Sent UDP (Lerp): {message_str}")
                
            except Exception as e:
                print(f"UDP Error: {e}")
        
        # Maintain 60Hz loop
        elapsed = time.time() - start_time
        sleep_time = interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

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
            crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # 3. Bilateral Filter (Edge-preserving smoothing)
        crop = cv2.bilateralFilter(crop, d=5, sigmaColor=75, sigmaSpace=75)
        
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
    # Nose tip: 1, Chin: 152, Left Eye Left Corner: 33, Right Eye Right Corner: 263, Left Mouth Corner: 61, Right Mouth Corner: 291
    image_points = np.array([
        (face_landmarks[1].x * w, face_landmarks[1].y * h),     # Nose tip
        (face_landmarks[152].x * w, face_landmarks[152].y * h), # Chin
        (face_landmarks[33].x * w, face_landmarks[33].y * h),   # Left eye left corner
        (face_landmarks[263].x * w, face_landmarks[263].y * h), # Right eye right corner
        (face_landmarks[61].x * w, face_landmarks[61].y * h),   # Left Mouth corner
        (face_landmarks[291].x * w, face_landmarks[291].y * h)  # Right mouth corner
    ], dtype="double")

    # 3D 模型点 (通用人脸模型)
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ], dtype="double")
    
    # Correct model points to match image points count (6 points)
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ], dtype="double")

    # PnP 求解
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    if not success:
        return 0, 0, 0

    # 计算欧拉角
    rmat, jac = cv2.Rodrigues(rotation_vector)
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
    
    # angles[0]=pitch, angles[1]=yaw, angles[2]=roll
    return angles[0], angles[1], angles[2]

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

# 打开摄像头，尝试不同的API
cap = None
for api in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
    cap = cv2.VideoCapture(0, api)
    if cap.isOpened():
        print(f"Using API: {api}")
        break

if not cap or not cap.isOpened():
    print("Error: Could not open camera")
    exit()

# 设置摄像头参数
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Camera Matrix for PnP
frame_w = 640
frame_h = 480
fov = 60.0
fov_rad = math.radians(fov)
focal_length = (frame_w / 2) / math.tan(fov_rad / 2)
cam_matrix = np.array([
    [focal_length, 0, frame_w / 2],
    [0, focal_length, frame_h / 2],
    [0, 0, 1]
], dtype="double")
dist_coeffs = np.zeros((4, 1))

# MediaPipe Iris Indices
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

# Start UDP Sender Thread
sender_thread = threading.Thread(target=udp_sender_thread, daemon=True)
sender_thread.start()

while True:
    # 读取帧
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break
    
    # print("Frame read successfully") # Debug

    
    # 翻转帧（可选，取决于摄像头是否镜像，这里保持原样以符合用户之前的逻辑）
    # frame = cv2.flip(frame, 1)
    
    h, w = frame.shape[:2]
    
    # 预处理：ROI -> 放大 -> 滤波 -> 增强
    processed_frame, roi_info = preprocessor.process(frame, last_landmarks_norm)
    
    # 转换处理后的帧为 RGB
    processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    
    # MediaPipe 处理
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=processed_rgb)
    timestamp_ms = int(time.time() * 1000)
    detection_result = detector.detect_for_video(mp_image, timestamp_ms)
    
    # 坐标还原 (Local Normalized -> Global Normalized)
    preprocessor.restore_landmarks(detection_result, roi_info, w, h)
    
    # 更新上一帧 Landmarks (用于下一帧 ROI 计算)
    if detection_result.face_landmarks:
        last_landmarks_norm = detection_result.face_landmarks[0]
    else:
        last_landmarks_norm = None
        
    # 可视化 ROI (蓝色矩形)
    rx, ry, rw, rh, _ = roi_info
    if rw < w or rh < h:
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
            
            # 绘制虹膜中心 (使用滤波后的坐标绘制，以反馈真实追踪位置)
            f_p1, f_p2 = eye_points
            cv2.circle(frame, (int(f_p1[0]), int(f_p1[1])), 3, (0, 255, 0), -1, cv2.LINE_AA) # Green for filtered
            cv2.circle(frame, (int(f_p2[0]), int(f_p2[1])), 3, (0, 255, 0), -1, cv2.LINE_AA)
            
            # 绘制原始点作为对比 (红色)
            cv2.circle(frame, (cx_left, cy_left), 2, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (cx_right, cy_right), 2, (0, 0, 255), -1, cv2.LINE_AA)
            
            # 计算距离和偏移 (使用滤波后的坐标)
            pixel_dist, estimated_dist = calculate_distance(eye_points, w, h)
            
            # 计算头部姿态并校正距离
            pitch, yaw, roll = get_head_pose(face_landmarks, w, h, cam_matrix, dist_coeffs)
            
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

            # Update buffer for UDP upsampling thread
            with data_lock:
                data_buffer.append((time.time(), tracker.current_estimated_dist, tracker.current_offset_x, tracker.current_offset_y))
    else:
        # 如果未检测到人脸，重置滤波器状态以便下次快速收敛
        tracker.reset()

    # --- 绘制信息 ---
    
    # 绘制光轴中心（十字准星）
    center_x, center_y = w // 2, h // 2
    cv2.line(frame, (center_x - 10, center_y), (center_x + 10, center_y), (0, 0, 255), 1)
    cv2.line(frame, (center_x, center_y - 10), (center_x, center_y + 10), (0, 0, 255), 1)
    
    # 绘制主视眼标记（黄色圆圈 + 'R'）
    if tracker.dominant_eye_pos is not None and detection_result.face_landmarks:
        dx, dy = tracker.dominant_eye_pos
        if 0 <= dx < w and 0 <= dy < h:
            cv2.circle(frame, (dx, dy), 8, (0, 255, 255), 2)
            cv2.putText(frame, "R", (dx + 10, dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    if tracker.current_pixel_dist > 0:
        if tracker.current_estimated_dist > 200:
            info_text = "too far!"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            info_text = f"Dist: {int(tracker.current_estimated_dist)}cm | PD: {int(tracker.current_pixel_dist)}px"
            offset_text = f"Offset X: {int(tracker.current_offset_x):+d}cm | Y: {int(tracker.current_offset_y):+d}cm"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, offset_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
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
is_running = False
cap.release()
cv2.destroyAllWindows()
