
import numpy as np

# Visualization
VISUALIZE = True # 设为 False 时，跳过所有可视化绘制，只保留计算和 UDP 发送

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
