
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

# --- Derived Model Points (Do Not Edit Directly) ---
# 构建 3D 坐标系 (Unit: cm): 
# Origin (0,0,0) at Nose Tip (Landmark 1)
# X+ Right (User's left), Y+ Up (User's Up), Z+ Forward (out of face)
# 注意：OpenCV Camera Frame 是 Y Down，Z Forward。
# 为了保持 PnP 结果的一致性，我们将 Model 建立为 Y Up (符合直觉)，
# 这样 PnP 得到的 rvec 会包含一个绕 X 轴的翻转 (180度)。
# 或者，我们可以建立 Y Down 的 Model，这样 rvec 就是 0 附近的微小旋转。
# 现有的代码似乎假设了 Model Y Up (因为 _y_eye 是正数，且 eyes 在 nose 上方)。
# 我们保持这个约定。

# 4-Point Model: Nose Tip, Brow Center, Left Eye Outer, Right Eye Outer
# 坐标单位：cm

# 1. Nose Tip (Landmark 1)
p_nose = (0.0, 0.0, 0.0)

# 2. Brow Center (Landmark 168)
# 位于鼻尖上方 FACE_REF_LENGTH_CM
# 深度：眉骨通常比鼻尖靠后，设为 -2.0 cm
p_brow = (0.0, FACE_REF_LENGTH_CM, -2.0)

# 3. Eyes (Landmark 33, 263)
# Y轴高度：通常位于鼻尖和眉心之间。
# 假设眼睛位于眉心下方 25% 处，或者鼻尖上方 75% 处？
# 简单起见，设为 FACE_REF_LENGTH_CM * 0.6
eye_y = FACE_REF_LENGTH_CM * 0.6
# X轴宽度：FACE_REF_WIDTH_CM / 2
eye_x = FACE_REF_WIDTH_CM / 2.0
# 深度：眼球比眉骨更深，设为 -2.5 cm (比鼻尖靠后 2.5cm)
eye_z = -2.5

p_eye_l = (-eye_x, eye_y, eye_z) # Left Eye (User's Left is Model X Negative?) 
# Wait, X+ is Right. User's Left is on the Right side of the image (if mirrored) or Left side?
# Usually Model X+ is User's Left (Stage Right).
# Let's check previous code: (-_x_eye_outer, _y_eye, _z_eye) was comment "Left eye outer".
# So X negative is Left Eye.
p_eye_r = (eye_x, eye_y, eye_z)  # Right Eye

MODEL_POINTS = np.array([
    p_nose,   # 1
    p_brow,   # 168
    p_eye_l,  # 33
    p_eye_r   # 263
], dtype="double")

# Eye Centers in Model Space (cm)
# 眼球中心比外眼角更靠内 (~1.5cm)，更深 (~1.2cm)
eye_center_x_offset = 1.5
eye_center_z_offset = 1.2

_x_eye_center = eye_x - eye_center_x_offset
_z_eye_center = eye_z - eye_center_z_offset

LEFT_EYE_CENTER_MODEL = np.array([-_x_eye_center, eye_y, _z_eye_center])
RIGHT_EYE_CENTER_MODEL = np.array([_x_eye_center, eye_y, _z_eye_center])

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
PREPROCESS_TARGET_HEIGHT = 1080  # 预处理目标高度
PREPROCESS_ROI_SCALE_FACTOR = 1.0 # ROI 区域二次缩放比例
PREPROCESS_GAUSSIAN_KERNEL_SIZE = (3, 3) # 高斯模糊核大小
PREPROCESS_GAUSSIAN_SIGMA = 0 # 高斯模糊 Sigma