
import os
import numpy as np

# Visualization
VISUALIZE = True # 设为 False 时，跳过所有可视化绘制，只保留计算和 UDP 发送

# Tracking Intervals (每多少帧处理一次)
# 设置为 1 表示每一帧都处理
EYE_TRACKING_INTERVAL = 1
HAND_TRACKING_INTERVAL = 1 # 手部追踪频率 (每多少帧进行一次手部检测)
EYE_GAZE_CALCULATION_INTERVAL = 2 # 视线解算频率 (每多少帧进行一次视线解算与绘制)
GAZE_RENDER_INTERVAL = 2 # 视线线段渲染频率 (每多少帧更新一次)
FULL_SCAN_INTERVAL = 6 # 全图扫描频率 (每多少帧进行一次全图扫描，如果 ROI 丢失)

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

# 手部 Yaw 角 OneEuroFilter 参数
HAND_YAW_ONE_EURO_MIN_CUTOFF = 1.0
HAND_YAW_ONE_EURO_BETA = 0.0
HAND_YAW_ONE_EURO_D_CUTOFF = 1.0

# 人脸距离一元滤波参数
FACE_DIST_ONE_EURO_MIN_CUTOFF = 0.3
FACE_DIST_ONE_EURO_BETA = 0.3
FACE_DIST_ONE_EURO_D_CUTOFF = 1.0

# 人脸位置一元滤波参数
FACE_POS_ONE_EURO_MIN_CUTOFF = 1.0
FACE_POS_ONE_EURO_BETA = 0.0
FACE_POS_ONE_EURO_D_CUTOFF = 1.0

# 头部 Yaw 角 OneEuroFilter 参数
FACE_YAW_ONE_EURO_MIN_CUTOFF = 1.0
FACE_YAW_ONE_EURO_BETA = 0.0
FACE_YAW_ONE_EURO_D_CUTOFF = 1.0

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

# --- 3D Face Model & Physical Parameters ---
# 用户可配置的真实面部物理参数 (单位: mm)
# 用于构建 solvePnP 的 3D 模型点
# 调整这些参数可以提高头部姿态解算的准确性

# 1. 眼部参数
P_OUTER_EYE_DIST_MM = 90.0   # 双眼外眼角间距 (标准值: ~90mm)
P_INNER_EYE_DIST_MM = 40.0   # 双眼内眼角间距 (标准值: ~30-35mm) - 用于辅助参考

# 2. 垂直距离参数 (相对于鼻尖)
P_NOSE_TO_CHIN_MM = 80.0     # 鼻尖到下巴尖的垂直距离
P_NOSE_TO_EYE_Y_MM = 45.0    # 鼻尖到眼睛中心线的垂直高度
P_NOSE_TO_MOUTH_Y_MM = 40.0  # 鼻尖到嘴角的垂直距离

# 3. 深度/前后参数 (相对于鼻尖，鼻尖 Z=0，面部其他部分 Z<0)
P_EYE_Z_OFFSET_MM = 25.0     # 眼睛所在的深度平面 (后缩)
P_MOUTH_Z_OFFSET_MM = 25.0   # 嘴角所在的深度平面
P_CHIN_Z_OFFSET_MM = 30.0    # 下巴尖所在的深度平面

# 4. 口部宽度
P_MOUTH_WIDTH_MM = 50.0      # 嘴角间距

# --- 3D Hand Model & Physical Parameters ---
# 用户可配置的真实手部物理参数 (单位: mm)
# 用于构建 solvePnP 的 3D 手部模型点
# 坐标系定义: 以手腕(Wrist)为原点
# 默认右手模型: 手心朝向相机，手指向上
# X轴: 水平向右 (拇指方向)
# Y轴: 垂直向上 (手指方向)
# Z轴: 垂直手掌向外 (指向相机)

# 手腕到各指根关节(MCP)的距离估计
# Index (食指), Middle (中指), Ring (无名指), Pinky (小指)

# 垂直高度 (Y轴)
P_WRIST_TO_INDEX_MCP_Y_MM = 95.0
P_WRIST_TO_MIDDLE_MCP_Y_MM = 95.0
P_WRIST_TO_RING_MCP_Y_MM = 90.0
P_WRIST_TO_PINKY_MCP_Y_MM = 80.0

# 水平偏移 (X轴) - 相对于手腕中心
# 假设右手手心朝前: 小指在左(-), 食指/拇指在右(+)
P_WRIST_TO_INDEX_MCP_X_MM = 25.0
P_WRIST_TO_MIDDLE_MCP_X_MM = 5.0
P_WRIST_TO_RING_MCP_X_MM = -10.0
P_WRIST_TO_PINKY_MCP_X_MM = -30.0

# 深度偏移 (Z轴) - 假设 MCP 关节稍微前倾或在一个平面
P_HAND_MCP_Z_OFFSET_MM = 0.0

# --- Derived Model Points (Do Not Edit Directly) ---
# 构建 3D 坐标系: 
# Origin (0,0,0) at Nose Tip
# X+ Right (User's left), Y+ Up, Z+ Forward (out of face)
# 之前的单位是 arbitrary units (approx 50 units/cm -> scale 5.0)
# 现在我们统一使用 mm 作为单位，或者保持之前的比例
# OpenCV solvePnP 的单位只要和相机内参一致即可 (通常相机内参基于像素，tvec 结果基于模型单位)
# 为了兼容现有逻辑 (tvec 结果单位)，我们使用 scale factor 将 mm 转换为之前的 "Model Units"
# 之前的 scale: Outer Eye Dist ~450 units / 9cm = 50 units/cm = 5 units/mm
MODEL_SCALE = 5.0 

# 计算模型点坐标
_x_eye_outer = (P_OUTER_EYE_DIST_MM / 2.0) * MODEL_SCALE
_y_eye = P_NOSE_TO_EYE_Y_MM * MODEL_SCALE
_z_eye = -P_EYE_Z_OFFSET_MM * MODEL_SCALE

_x_mouth = (P_MOUTH_WIDTH_MM / 2.0) * MODEL_SCALE
_y_mouth = -P_NOSE_TO_MOUTH_Y_MM * MODEL_SCALE
_z_mouth = -P_MOUTH_Z_OFFSET_MM * MODEL_SCALE

_y_chin = -P_NOSE_TO_CHIN_MM * MODEL_SCALE
_z_chin = -P_CHIN_Z_OFFSET_MM * MODEL_SCALE

MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),                  # Nose tip
    (0.0, _y_chin, _z_chin),          # Chin
    (-_x_eye_outer, _y_eye, _z_eye),  # Left eye outer (Model X is negative for Left eye if X+ is Right)
    (_x_eye_outer, _y_eye, _z_eye),   # Right eye outer
    (-_x_mouth, _y_mouth, _z_mouth),  # Left mouth corner
    (_x_mouth, _y_mouth, _z_mouth)    # Right mouth corner
], dtype="double")

# --- Derived Hand Model Points ---
# 1. 缩放: 将 mm 转换为 Model Units
_hand_idx_x = P_WRIST_TO_INDEX_MCP_X_MM * MODEL_SCALE
_hand_idx_y = P_WRIST_TO_INDEX_MCP_Y_MM * MODEL_SCALE

_hand_mid_x = P_WRIST_TO_MIDDLE_MCP_X_MM * MODEL_SCALE
_hand_mid_y = P_WRIST_TO_MIDDLE_MCP_Y_MM * MODEL_SCALE

_hand_ring_x = P_WRIST_TO_RING_MCP_X_MM * MODEL_SCALE
_hand_ring_y = P_WRIST_TO_RING_MCP_Y_MM * MODEL_SCALE

_hand_pinky_x = P_WRIST_TO_PINKY_MCP_X_MM * MODEL_SCALE
_hand_pinky_y = P_WRIST_TO_PINKY_MCP_Y_MM * MODEL_SCALE

_hand_mcp_z = P_HAND_MCP_Z_OFFSET_MM * MODEL_SCALE

# 2. 构建 3D 点集 (用于 solvePnP)
# 对应 MediaPipe Hands 关键点索引:
# 0: Wrist
# 5: Index Finger MCP
# 9: Middle Finger MCP
# 13: Ring Finger MCP
# 17: Pinky MCP

HAND_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),                      # 0: Wrist
    (_hand_idx_x, _hand_idx_y, _hand_mcp_z),     # 5: Index MCP
    (_hand_mid_x, _hand_mid_y, _hand_mcp_z),     # 9: Middle MCP
    (_hand_ring_x, _hand_ring_y, _hand_mcp_z),   # 13: Ring MCP
    (_hand_pinky_x, _hand_pinky_y, _hand_mcp_z)  # 17: Pinky MCP
], dtype="double")

# 对应的关键点索引 (用于从检测结果中提取 2D 点)
HAND_MODEL_INDICES = [0, 5, 9, 13, 17]

# Eye Centers in Model Space
# 估算眼球中心位置 (比外眼角更靠内，深度更深)
# 假设眼球中心位于外眼角内侧 ~15mm, 深度再深 ~12mm
_x_eye_center = _x_eye_outer - (15.0 * MODEL_SCALE)
_z_eye_center = _z_eye - (12.0 * MODEL_SCALE)

LEFT_EYE_CENTER_MODEL = np.array([-_x_eye_center, _y_eye, _z_eye_center])
RIGHT_EYE_CENTER_MODEL = np.array([_x_eye_center, _y_eye, _z_eye_center])

# Eye Ball Radius (12mm)
EYE_RADIUS = 12.0 * MODEL_SCALE

# Screen Projection
AXIS_LENGTH = 100.0 * MODEL_SCALE # 10cm line

# --- Physical Constants (Unit: cm / meters as specified) ---

# Eye Distance Constants (cm)
# Used for depth estimation based on eye corner distances
# Sync with PnP parameters for consistency
INNER_EYE_DIST_CM = P_INNER_EYE_DIST_MM / 10.0
OUTER_EYE_DIST_CM = P_OUTER_EYE_DIST_MM / 10.0

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
FACE_MESH_TASK_PATH = os.path.join('models', 'face_landmarker.task')
HAND_LANDMARKER_TASK_PATH = os.path.join('models', 'hand_landmarker.task')

# --- Image Preprocessing ---
PREPROCESS_TARGET_HEIGHT = 1080  # 预处理目标高度
PREPROCESS_ROI_SCALE_FACTOR = 1.0 # ROI 区域二次缩放比例
PREPROCESS_GAUSSIAN_KERNEL_SIZE = (5, 5) # 高斯模糊核大小
PREPROCESS_GAUSSIAN_SIGMA = 0 # 高斯模糊 Sigma