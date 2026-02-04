import cv2
import numpy as np
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import socket
import json

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
        
        # Initialize Kalman filters for distance smoothing
        self.pixel_dist_filter = OneDKalmanFilter(Q=0.1, R=5.0)
        self.real_dist_filter = OneDKalmanFilter(Q=0.1, R=5.0)
        # Filters for offset
        self.offset_x_filter = OneDKalmanFilter(Q=0.1, R=0.1)
        self.offset_y_filter = OneDKalmanFilter(Q=0.1, R=0.1)
        # 记录主视眼位置用于绘制
        self.dominant_eye_pos = None
    
    def reset(self):
        self.dominant_eye_pos = None
        # Keep current distance values until new ones are calculated to avoid flickering
    
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
        self.dominant_eye_pos = right_eye
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
        
        # 滤波
        filtered_x = self.offset_x_filter.update(real_offset_x)
        filtered_y = self.offset_y_filter.update(real_offset_y)
        
        # 保留浮点数精度
        self.current_offset_x = filtered_x
        self.current_offset_y = filtered_y

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

# Initialize UDP Socket
UDP_IP = "127.0.0.1"
UDP_PORT = 8888
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

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
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # MediaPipe 处理
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int(time.time() * 1000)
    detection_result = detector.detect_for_video(mp_image, timestamp_ms)
    
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
            
            eye_points = [(cx_left, cy_left), (cx_right, cy_right)]
            
            # 绘制虹膜中心
            cv2.circle(frame, (cx_left, cy_left), 3, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (cx_right, cy_right), 3, (0, 0, 255), -1, cv2.LINE_AA)
            
            # 绘制人脸网格（可选，仅绘制眼睛轮廓）
            # 这里简单绘制两个点即可
            
            # 计算距离和偏移
            pixel_dist, estimated_dist = calculate_distance(eye_points, w, h)
            
            # 计算头部姿态并校正距离
            pitch, yaw, roll = get_head_pose(face_landmarks, w, h, cam_matrix, dist_coeffs)
            
            # 三角变换校正: 
            # 投影瞳距 = 真实瞳距 * cos(Yaw)
            # 估算距离 = (焦距 * 真实瞳距) / 投影瞳距 = (焦距 * 真实瞳距) / (真实瞳距像素 * cos(Yaw))
            #          = (正面估算距离) / cos(Yaw)
            # 因此: 正面估算距离 = 当前估算距离 * cos(Yaw)
            correction_factor = math.cos(math.radians(yaw))
            if correction_factor < 0.2: correction_factor = 0.2 # 防止极端角度导致数值异常
            
            corrected_dist = estimated_dist * correction_factor
            
            # 应用滤波
            filtered_pixel, filtered_estimated = tracker.apply_distance_filter(pixel_dist, corrected_dist)
            tracker.current_pixel_dist = filtered_pixel
            tracker.current_estimated_dist = filtered_estimated
            
            # 更新偏移量
            tracker.update_offset(eye_points, w, h, filtered_pixel, filtered_estimated)

            # Send data via UDP
            # Format: distance,offset_x,offset_y (保留1位小数)
            message_str = f"{tracker.current_estimated_dist:.1f},{tracker.current_offset_x:.1f},{tracker.current_offset_y:.1f}"
            try:
                sock.sendto(message_str.encode('utf-8'), (UDP_IP, UDP_PORT))
                print(f"Sent UDP: {message_str}")
            except Exception as e:
                print(f"UDP Error: {e}")
    else:
        # 如果未检测到人脸，可以重置或保持最后状态
        pass

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
            
            # 绘制瞳孔间距线
            if len(eye_points) >= 2:
                cv2.line(frame, eye_points[0], eye_points[1], (255, 255, 0), 2)

    # 显示结果
    cv2.imshow('Face and Eye Detection (MediaPipe)', frame)
    
    # 按ESC键退出
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
