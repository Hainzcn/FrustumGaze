
import numpy as np

# Visualization
VISUALIZE = True # 设为 False 时，跳过所有可视化绘制，只保留计算和 UDP 发送

# Tracking Intervals (每多少帧处理一次)
# 设置为 1 表示每一帧都处理
EYE_TRACKING_INTERVAL = 1
HAND_TRACKING_INTERVAL = 1
GAZE_RENDER_INTERVAL = 1 # 视线线段渲染频率 (每多少帧更新一次)
HAND_FULL_SCAN_INTERVAL = 6 # 全图扫描频率 (每多少帧进行一次全图扫描，如果 ROI 丢失)

# 滤波器参数设置

# 手部距离一元滤波参数
HAND_DIST_ONE_EURO_MIN_CUTOFF = 0.5
HAND_DIST_ONE_EURO_BETA = 0.2
HAND_DIST_ONE_EURO_D_CUTOFF = 1.0

# Hand OneEuroFilter Parameters (Position/PnP - Landmarks)
HAND_POS_ONE_EURO_MIN_CUTOFF = 1.0
HAND_POS_ONE_EURO_BETA = 0.0
HAND_POS_ONE_EURO_D_CUTOFF = 1.0

# Hand Kalman Filter Parameters (for 3D position smoothing)
HAND_KALMAN_PROCESS_NOISE = 0.01    # Q: Process noise covariance (trust in model)
HAND_KALMAN_MEASUREMENT_NOISE = 0.1 # R: Measurement noise covariance (trust in measurement)

# 人脸距离一元滤波参数
FACE_DIST_ONE_EURO_MIN_CUTOFF = 0.3
FACE_DIST_ONE_EURO_BETA = 0.3
FACE_DIST_ONE_EURO_D_CUTOFF = 1.0

# 人脸位置一元滤波参数
FACE_POS_ONE_EURO_MIN_CUTOFF = 1.0
FACE_POS_ONE_EURO_BETA = 0.0
FACE_POS_ONE_EURO_D_CUTOFF = 1.0

# 人脸距离卡尔曼滤波参数
FACE_DIST_KALMAN_Q = 0.1 # Process Noise (Q)
FACE_DIST_KALMAN_R = 5.0 # Measurement Noise (R)

# 人脸位置偏移卡尔曼滤波参数
FACE_OFFSET_KALMAN_Q = 0.3 # Process Noise (Q)
FACE_OFFSET_KALMAN_R = 0.1 # Measurement Noise (R)

# Network
UDP_IP = "127.0.0.1"
UDP_PORT = 8888

# MediaPipe Iris Indices
# 网格点 468 是左虹膜中心 (用户左侧)
# 网格点 473 是右虹膜中心 (用户右侧)
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

# 3D Face Model Points (General Face Model, approx 50 units/cm)
# Coordinate System: X Right, Y Up, Z Forward (relative to face center)
MODEL_POINTS = np.array([
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

# Eye Centers in Model Space
# Z axis offset 12mm (60 units) into the skull
LEFT_EYE_CENTER_MODEL = np.array([-157.5, 170.0, -187.5])
RIGHT_EYE_CENTER_MODEL = np.array([157.5, 170.0, -187.5])

# Eye Ball Radius in Model Units
EYE_RADIUS = 60.0

# Screen Projection
AXIS_LENGTH = 500.0

# --- Physical Constants (Unit: cm / meters as specified) ---

# Eye Distance Constants (cm)
# Used for depth estimation based on eye corner distances
INNER_EYE_DIST_CM = 4.0
OUTER_EYE_DIST_CM = 9.0

# Hand Constants
HAND_PALM_WIDTH_CM = 6.0  # Default palm width (distance between index and pinky MCP)
PINCH_THRESHOLD_M = 0.02  # 2cm threshold for pinch detection

# --- Tracking Confidence Thresholds ---

# Face Tracking Confidence
FACE_MIN_DETECTION_CONFIDENCE = 0.5
FACE_MIN_PRESENCE_CONFIDENCE = 0.5
FACE_MIN_TRACKING_CONFIDENCE = 0.5

# Hand Tracking Confidence
HAND_MIN_DETECTION_CONFIDENCE = 0.5
HAND_MIN_PRESENCE_CONFIDENCE = 0.5
HAND_MIN_TRACKING_CONFIDENCE = 0.5

# Model Paths
FACE_MESH_TASK_PATH = 'face_landmarker.task'
HAND_LANDMARKER_TASK_PATH = 'hand_landmarker.task'

# --- Image Preprocessing ---
PREPROCESS_TARGET_HEIGHT = 720  # 预处理目标高度
PREPROCESS_ROI_SCALE_FACTOR = 0.5 # ROI 区域二次缩放比例
PREPROCESS_GAUSSIAN_KERNEL_SIZE = (5, 5) # 高斯模糊核大小
PREPROCESS_GAUSSIAN_SIGMA = 0 # 高斯模糊 Sigma