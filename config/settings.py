
import os
import numpy as np

# Visualization
VISUALIZE = True # 设为 False 时，跳过所有可视化绘制，只保留计算和 UDP 发送

# Tracking Intervals (每多少帧处理一次)
# 设置为 1 表示每一帧都处理
EYE_TRACKING_INTERVAL = 1
HAND_TRACKING_INTERVAL = 1 # 手部追踪频率 (每多少帧进行一次手部检测)
POSE_TRACKING_INTERVAL = 6 # 姿态追踪频率 (每多少帧进行一次姿态检测)
EYE_GAZE_CALCULATION_INTERVAL = 3 # 视线解算频率 (每多少帧进行一次视线解算与绘制)
GAZE_RENDER_INTERVAL = 3 # 视线线段渲染频率 (每多少帧更新一次)
FULL_SCAN_INTERVAL = 6 # 全图扫描频率 (每多少帧进行一次全图扫描，如果 ROI 丢失)

# 滤波器参数配置 (Hierarchical Filter Configuration)
FILTER_CONFIG = {
    # Level 1: 关键点滤波 (Shared Keypoint Filtering)
    # 用于平滑 MediaPipe 原始输出坐标 (Normalized x, y, z)
    'KEYPOINT': {
        'min_cutoff': 1.0,  # 默认截止频率 (Hz) - 较高值减少延迟
        'beta': 0.5,        # 速度系数 - 较高值在快速运动时减少延迟
        'd_cutoff': 1.0     # 导数截止频率 (Hz)
    },

    # Level 2 & 3: 高级数据滤波 (High-Level Data Filtering)
    'HAND': {
        # 3D 空间位置 (Kalman Filter)
        'POSITION': {
            'process_noise': 0.01,     # Q: 过程噪声 (信任模型)
            'measurement_noise': 0.1,  # R: 测量噪声 (信任测量)
            'r_grip_max': 1.0,         # 握拳时的最大附加测量噪声
        },
        # 归一化宽度/尺度 (OneEuro)
        'SCALE': {
            'min_cutoff': 0.5,
            'beta': 0.2,
            'd_cutoff': 1.0
        },
        # Yaw 角度 (OneEuro)
        'YAW': {
            'min_cutoff': 1.0,
            'beta': 0.0,
            'd_cutoff': 1.0
        },
        # Pitch 角度 (OneEuro)
        'PITCH': {
            'min_cutoff': 1.0,
            'beta': 0.0,
            'd_cutoff': 1.0
        },
        # 像素距离 (Pixel Distance) (OneEuro)
        'PIXEL_DIST': {
            'min_cutoff': 1.0,  # 较高以保持对快速运动的响应
            'beta': 0.2,        # 适度平滑
            'd_cutoff': 1.0
        },
        # 深度 Z 值 (OneEuro - Pre-XY Calculation)
        'Z_VAL': {
            'min_cutoff': 0.5,  # 较低以获得更稳定的 Z
            'beta': 0.1,
            'd_cutoff': 1.0
        },
        # 深度动态参数
        'DEPTH': {
            'history_size': 15,
            'sigma_threshold_ratio': 0.03,
            'anchor_yaw_threshold': 15.0,
            'anchor_grip_threshold': 0.2,
            'anchor_halflife_frames': 45,
            'grip_smoothing_alpha': 0.3,
            'grip_ref_scale': 1.2
        }
    },

    'FACE': {
        # 距离估算 (Kalman Filter)
        'DISTANCE': {
            'process_noise': 0.1,  # Q
            'measurement_noise': 5.0 # R
        },
        # 中心偏移量 (Kalman Filter)
        'OFFSET': {
            'process_noise': 0.3,  # Q
            'measurement_noise': 0.1 # R
        },
        # Yaw 角度 (OneEuro)
        'YAW': {
            'min_cutoff': 1.0,
            'beta': 0.0,
            'd_cutoff': 1.0
        },
        # 虹膜/眼球中心 (OneEuro - 通常需要更平滑)
        'IRIS': {
            'min_cutoff': 0.5,
            'beta': 0.1,
            'd_cutoff': 1.0
        }
    }
}

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
HAND_REF_LENGTH_M = 0.09  # Reference length (Wrist to Middle MCP) for depth estimation. 9cm.
HAND_REF_WIDTH_M = 0.06 # Reference width (Index MCP to Pinky MCP) for depth estimation. 6cm.
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

POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_PRESENCE_CONFIDENCE = 0.5
POSE_MIN_TRACKING_CONFIDENCE = 0.5

# Model Paths
FACE_MESH_TASK_PATH = os.path.join('models', 'face_landmarker.task')
HAND_LANDMARKER_TASK_PATH = os.path.join('models', 'hand_landmarker.task')
POSE_LANDMARKER_TASK_PATH = os.path.join('models', 'pose_landmarker.task')

# --- Image Preprocessing ---
PREPROCESS_TARGET_HEIGHT = 1080  # 预处理目标高度
PREPROCESS_ROI_SCALE_FACTOR = 1.0 # ROI 区域二次缩放比例
PREPROCESS_GAUSSIAN_KERNEL_SIZE = (3, 3) # 高斯模糊核大小
PREPROCESS_GAUSSIAN_SIGMA = 0 # 高斯模糊 Sigma