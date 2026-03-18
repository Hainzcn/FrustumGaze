
import os
import numpy as np

# Visualization
VISUALIZE = True # 设为 False 时，跳过所有可视化绘制，只保留计算和 UDP 发送
PRINT_UDP_DATA = True # 设为 True 时，在终端打印 UDP 发送的数据

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
        # Pitch 角度 (OneEuro)
        'PITCH': {
            'min_cutoff': 1.0,
            'beta': 0.0,
            'd_cutoff': 1.0
        },
        # 虹膜/眼球中心 (OneEuro - 通常需要更平滑)
        'IRIS': {
            'min_cutoff': 0.5,
            'beta': 0.1,
            'd_cutoff': 1.0
        },
        # 深度 Z 值 (OneEuro)
        'Z_VAL': {
            'min_cutoff': 0.5,
            'beta': 0.1,
            'd_cutoff': 1.0
        },
        # 动态校准参数
        'CALIBRATION': {
            'width_correction_alpha': 0.05, # 校准平滑系数
            'min_valid_yaw': 15.0, # 校准时的最大偏航角
            'min_valid_pitch': 15.0 # 校准时的最大俯仰角
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
# 用户可配置的真实面部物理参数 (单位: cm)
# 用于构建 solvePnP 的 3D 模型点

# Face Constants
FACE_REF_LENGTH_CM = 8.0  # 眉心到鼻尖的垂直距离 (参考值)
FACE_REF_WIDTH_CM = 9.0   # 双眼外眼角间距 (参考值)
FACE_REF_MOUTH_WIDTH_CM = 5.0 # 嘴角间距 (参考值)
FACE_REF_MOUTH_DOWN_CM = 4.0 # 鼻尖到嘴角的垂直距离 (参考值)

# --- Derived Model Points (Do Not Edit Directly) ---
# 3D Coordinate System (Unit: cm):
# Origin: Nose Tip (0,0,0)
# X+: User's Left
# Y+: Up
# Z+: Forward (out of face)

# Eye Centers in Model Space (cm)
# Used for Gaze Calculation relative to Head Center/Nose
# X axis: + is Left, - is Right
# Y axis: + is Up
# Z axis: + is Forward

# Reference dimensions
eye_y = FACE_REF_LENGTH_CM * 0.6
eye_x = FACE_REF_WIDTH_CM / 2.0
eye_z = -2.5

eye_center_x_offset = 1.5
eye_center_z_offset = 1.2

_x_eye_center = eye_x - eye_center_x_offset
_z_eye_center = eye_z - eye_center_z_offset

LEFT_EYE_CENTER_MODEL = np.array([_x_eye_center, eye_y, _z_eye_center])
RIGHT_EYE_CENTER_MODEL = np.array([-_x_eye_center, eye_y, _z_eye_center])

# Eye Ball Radius (1.2cm)
EYE_RADIUS = 1.2

# Screen Projection Axis Length (10cm)
AXIS_LENGTH = 10.0

# --- Physical Constants (Unit: cm / meters as specified) ---

# Eye Distance Constants (cm)
# Used for depth estimation based on eye corner distances
# Sync with PnP parameters for consistency
INNER_EYE_DIST_CM = 4.0 # Keep as fallback
OUTER_EYE_DIST_CM = FACE_REF_WIDTH_CM

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
PREPROCESS_TARGET_HEIGHT = 1080  # 预处理目标高度 (Face / Hand)
POSE_TARGET_HEIGHT = 360         # Pose 独立目标高度 (仅需 4 个粗关键点，360p 足矣)
PREPROCESS_ROI_SCALE_FACTOR = 1.0 # ROI 区域二次缩放比例
PREPROCESS_GAUSSIAN_KERNEL_SIZE = (3, 3) # 高斯模糊核大小
PREPROCESS_GAUSSIAN_SIGMA = 0 # 高斯模糊 Sigma