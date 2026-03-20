# API 参考文档

本文档主要介绍 FrustumGaze 项目中核心类和模块的接口说明。

## 1. 核心管线 (modules.pipeline)

### `FrustumGazePipeline` (in `modules/pipeline/manager.py`)

主程序入口类，负责协调各模块工作。

**主要方法:**

*   `__init__()`: 初始化队列、同步事件、`StatsManager`、`UDPSender`、`Visualizer` 等核心管理器。
*   `setup() -> bool`: 
    *   自动选择摄像头 API 后端（DSHOW, MSMF 等）。
    *   配置分辨率、曝光、MJPEG 编码。
    *   创建三缓冲共享内存。
    *   初始化 `CameraModel` 相机内参。
*   `start_processes()`: 创建并启动 `FrameProcessorProcess`、`HandProcessorProcess`、`PoseProcessorProcess` 子进程。
*   `run()`: 启动主循环——帧捕捉、任务分发、结果收集、统计更新、渲染。
*   `stop()`: 安全停止所有子进程、视频流，释放共享内存和 OpenCV 窗口。

**内部方法:**

*   `_process_frame()`: 从视频流获取帧，按 `*_TRACKING_INTERVAL` 频率向各子进程队列分发任务。
*   `_check_face_results()`: 非阻塞检查面部追踪输出队列，提取 `GazeResult`，通过 UDP 发送视线数据。
*   `_check_hand_results()`: 非阻塞检查手部追踪输出队列，提取手部位置和捏合状态，通过 UDP 发送。
*   `_check_pose_results()`: 非阻塞检查姿态追踪输出队列。
*   `_update_stats()`: 调用 `StatsManager.update_drop_rate()` 和 `update_resource_usage()`。
*   `_render() -> bool`: 调用 `Visualizer.render()` 显示追踪结果，返回是否按下 ESC。

### `BaseProcessorProcess` (in `modules/pipeline/base_process.py`)

子进程基类，封装共享内存连接、主循环、队列交互和生命周期管理。

**子类钩子:**

*   `on_init() -> bool`: 初始化子进程本地资源（tracker 等），返回 True 表示成功。
*   `on_process(task, frame) -> dict | None`: 处理单帧，返回结果 dict 放入输出队列，返回 None 跳过。
*   `on_cleanup()`: 释放子进程本地资源（可选）。

## 2. 追踪器 (trackers)

### `FrameProcessorProcess` (in `modules/pipeline/face_process.py`)

运行在独立进程中的人脸追踪器，继承 `BaseProcessorProcess`。

**输入任务结构:**
```python
{
    'frame_id': int,      # 帧序号
    'buffer_idx': int     # 共享内存索引
}
```

**输出结果结构:**
```python
{
    'detection_result': FaceLandmarkerResult,  # MediaPipe 原始结果
    'roi_info': tuple,          # (x, y, w, h, scale) 当前使用的 ROI
    'using_full_scan': bool,    # 是否正在全图扫描
    'gaze_result': GazeResult,  # 视线追踪结果 (仅 ROI 模式下)
}
```

### `HandProcessorProcess` (in `modules/pipeline/hand_process.py`)

运行在独立进程中的手部追踪器，继承 `BaseProcessorProcess`。

**输出结果结构:**
```python
{
    'hand_result': HandLandmarkerResult,  # MediaPipe 原始结果
    'closest_hand': dict,   # 距离最近的手的信息 (含 x, y, z, is_pinching 等)
    'hands_pos': list       # 所有检测到的手部位置信息
}
```

### `PoseProcessorProcess` (in `modules/pipeline/pose_process.py`)

运行在独立进程中的姿态追踪器，继承 `BaseProcessorProcess`。

**输出结果结构:**
```python
{
    'pose_result': PoseLandmarkerResult,  # MediaPipe 原始结果
}
```

### `GazeResult` (in `trackers/eye_tracker.py`)

子进程→主进程的视线追踪结果 dataclass，作为跨进程传输的数据契约。

**字段:**

| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `estimated_dist` | float | 头部深度估计值 (cm) |
| `offset_x` / `offset_y` | float | 头部中心相对光轴的物理偏移 (cm) |
| `pixel_dist` | float | 双眼外眼角像素间距 |
| `yaw` / `pitch` / `roll` | float | 头部欧拉角 (度) |
| `head_center_pos` | tuple | 头部中心的 2D 像素坐标 |
| `depth_details` | dict | 双通道深度融合详情 |
| `eye_points` / `raw_eye_points` | list | 滤波后/原始虹膜中心坐标 |
| `left_gaze_vec` / `right_gaze_vec` | np.ndarray | 左/右眼 3D 视线向量 |
| `left_eye_center_cam` / `right_eye_center_cam` | np.ndarray | 左/右眼球中心 3D 坐标 |
| `screen_point` | tuple | 视线与屏幕平面交点 |
| `left_confidence` / `right_confidence` | float | 左/右眼视线置信度 |
| `rmat` / `rvec` | np.ndarray | 旋转矩阵/向量 (仅用于可视化) |

## 3. 工具类 (utils)

### `ImagePreprocessor` (in `utils/image_utils.py`)

图像预处理工具，管理 ROI 裁剪、放大和对比度增强。

*   `process(frame, last_landmarks, padding_factor) -> (processed_frame, roi_info)`: ROI 裁剪 → 放大 → CLAHE 增强。
*   `restore_landmarks(detection_result, roi_info, w_frame, h_frame)`: 将 ROI 局部坐标还原为全图归一化坐标。

### `GlobalImagePreprocessor` (in `utils/image_utils.py`)

全局静态图像处理工具类，提供 CLAHE、高斯模糊、缩放、色域转换等静态方法。

*   `apply_clahe(image, clip_limit, tile_grid_size)`: 自适应直方图均衡化（CLAHE 实例按参数缓存复用）。
*   `resize_image(image, target_size, scale_factor)`: 统一缩放逻辑。
*   `to_rgb(image)` / `to_gray(image)` / `to_lab(image)`: 色域转换。

### `OneEuroFilter` (in `utils/math_utils.py`)

一欧元滤波器，用于处理高频抖动。核心计算可选使用 Numba JIT 加速。

*   `filter(x, t=None) -> float`: 输入新值，返回平滑后的值。
*   参数: `min_cutoff` (截止频率), `beta` (速度系数), `d_cutoff` (导数截止频率)。

### `Simple3DKalmanFilter` (in `utils/math_utils.py`)

基于 OpenCV KalmanFilter 的 3D 坐标平滑器（6 状态、3 观测）。

*   `update(x, y, z, R_z=None) -> (float, float, float)`: 更新观测值，可选动态调整 Z 轴测量噪声。

### `OneDKalmanFilter` (in `utils/math_utils.py`)

一维卡尔曼滤波器，用于距离/偏移等标量平滑。

*   `update(measurement) -> float`: 更新并返回滤波后的值。

### 辅助函数 (in `utils/math_utils.py`)

*   `calculate_screen_intersection(eye_pos, gaze_vec, z_plane=0.0)`: 计算视线与屏幕平面 (Z=z_plane) 的交点。
*   `calculate_weighted_average(p1, p2, w1, w2)`: 两个点/向量的加权平均。

## 4. 性能统计 (modules.stats)

### `StatsManager` (in `modules/stats.py`)

性能统计管理器，负责 FPS、丢包率、延迟及资源占用统计。

*   `update_fps() -> float`: 更新并返回滑动窗口 FPS 和 P99 延迟。
*   `record_captured()` / `record_face_task_attempted()` / `record_face_task_dropped()` 等: 记录帧/任务计数。
*   `update_drop_rate() -> float`: 每秒计算一次丢包率。
*   `update_resource_usage()`: 每秒采样一次进程 CPU / 内存（`psutil`）及 GPU 使用率（`GPUtil`）。依赖缺失时跳过。
*   `get_stats() -> dict`: 返回完整统计字典（fps, drop_rate, p99_latency, cpu_percent, mem_mb, gpu_util, gpu_mem_mb）。

## 5. 网络通信 (modules.network)

### `UDPSender` (in `modules/network.py`)

异步 UDP 发送器，使用后台线程和队列避免网络抖动阻塞主线程。

*   `__init__(ip, port, queue_size=10)`: 创建 UDP socket 并启动发送线程。
*   `send(data_str)`: 非阻塞发送字符串数据。队列满时丢弃最旧数据（实时数据丢弃策略）。
*   `close()`: 停止发送线程并关闭 socket。

## 6. 配置 (config)

### `Settings` (in `config/settings.py`)

全局静态配置参数。

**可视化与调试:**

*   `VISUALIZE`: bool, 是否开启可视化窗口。
*   `PRINT_UDP_DATA`: bool, 是否在终端打印 UDP 发送数据。

**追踪频率控制:**

*   `EYE_TRACKING_INTERVAL`: 人脸追踪降频系数。
*   `HAND_TRACKING_INTERVAL`: 手部追踪降频系数。
*   `POSE_TRACKING_INTERVAL`: 姿态追踪降频系数。
*   `EYE_GAZE_CALCULATION_INTERVAL`: 视线解算降频系数。
*   `GAZE_RENDER_INTERVAL`: 视线渲染降频系数。
*   `FULL_SCAN_INTERVAL`: 全图扫描频率。

**物理常量 (cm):**

*   `FACE_REF_LENGTH_CM` / `FACE_REF_WIDTH_CM`: 面部参考尺寸（眉心-鼻尖 / 双眼外眼角间距）。
*   `HAND_REF_LENGTH_CM` / `HAND_REF_WIDTH_CM`: 手部参考尺寸（腕-中指MCP / 食指MCP-小指MCP）。
*   `PINCH_THRESHOLD_CM` / `PINCH_DEBOUNCE_FRAMES`: 捏合检测阈值和去抖帧数。
*   `EYE_RADIUS`: 眼球半径 (1.2cm)。
*   `GAZE_CONFIDENCE_YAW_SENSITIVITY` / `GAZE_CONFIDENCE_MIN`: 视线置信度参数。

**网络:**

*   `UDP_IP` / `UDP_PORT`: UDP 目标地址和端口。

**模型路径:**

*   `FACE_MESH_TASK_PATH` / `HAND_LANDMARKER_TASK_PATH` / `POSE_LANDMARKER_TASK_PATH`

**滤波器配置:**

*   `FILTER_CONFIG`: 分层字典结构，包含 `KEYPOINT`（共享关键点滤波）、`HAND`（位置/尺度/深度等）、`FACE`（距离/偏移/角度/虹膜/校准等）三大类。详见 `config/settings.py` 源码。

### `ConfigManager` (in `modules/camera.py`)

管理用户偏好和摄像头配置（JSON 持久化）。

*   `get_camera_info(device_id) -> dict`: 获取指定摄像头的校准信息（曝光、分辨率等）。
*   `update_camera(device_id, **kwargs)`: 更新摄像头配置。
*   `get_last_camera() -> str`: 获取上次使用的摄像头设备 ID。

### `CameraModel` (in `modules/camera.py`)

摄像头内参模型，根据分辨率和 FOV 构建相机矩阵。

*   `cam_matrix`: 3x3 相机内参矩阵。
*   `dist_coeffs`: 畸变系数（默认全零）。
