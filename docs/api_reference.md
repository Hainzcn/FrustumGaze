# API 参考文档

本文档主要介绍 FrustumGaze 项目中核心类和模块的接口说明。

## 1. 核心管线 (modules.pipeline)

### `FrustumGazePipeline`

主程序入口类，负责协调各模块工作。

**主要方法:**

*   `__init__()`: 初始化资源，创建队列和同步锁。
*   `setup()`: 
    *   加载配置文件。
    *   初始化摄像头（尝试 DSHOW, MSMF 等后端）。
    *   初始化共享内存。
    *   启动子进程。
*   `run()`: 启动主循环，处理帧捕捉、任务分发、结果处理、渲染和发送。
*   `_process_frame()`: 内部方法，处理单帧逻辑。
*   `stop()`: 安全停止所有子进程，释放资源。

## 2. 追踪器 (trackers)

### `FaceMeshProcess` (in `modules/pipeline/face_process.py`)

运行在独立进程中的人脸追踪器。

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
    'frame_id': int,
    'face_result': NormalizedLandmarkList, # MediaPipe 原始结果
    'roi_info': tuple,    # (x, y, w, h) 当前使用的 ROI
    'timestamp': float    # 处理完成时间戳
}
```

### `HandProcessorProcess` (in `modules/pipeline/hand_process.py`)

运行在独立进程中的手部追踪器。

**输出结果结构:**
```python
{
    'frame_id': int,
    'hand_result': HandLandmarkerResult,
    'closest_hand': dict, # 距离最近的手的信息
    'hands_pos': list     # 所有检测到的手部位置信息
}
```

## 3. 工具类 (utils)

### `ImagePreprocessor` (in `utils/image_utils.py`)

图像预处理工具，主要用于 ROI 裁剪和坐标还原。

*   `get_roi(landmarks, frame_size)`: 根据关键点计算包围盒 ROI。
*   `recover_landmarks(landmarks, roi, frame_size)`: 将 ROI 局部坐标系下的关键点还原回全图坐标系。

### `OneEuroFilter` (in `utils/math_utils.py`)

一欧元滤波器，用于处理高频抖动。

*   `update(value, timestamp)`: 输入新值，返回平滑后的值。
*   `min_cutoff`: 最小截止频率，控制慢速运动时的平滑度。
*   `beta`: 速度系数，控制快速运动时的响应速度。

## 4. 网络通信 (modules.network)

### `UDPSender`

负责向 Unity 发送数据。

*   `send_packet(data_dict)`: 将字典数据序列化（通常为 JSON 或 struct）并发送。
*   `send_bytes(bytes_data)`: 发送原始字节流。

## 5. 配置 (config)

### `Settings` (in `config/settings.py`)

全局静态配置参数。

*   `VISUALIZE`: bool, 是否开启可视化窗口。
*   `UDP_IP`, `UDP_PORT`: 目标地址。
*   `EYE_TRACKING_INTERVAL`: 人脸追踪降频系数。
*   `HAND_TRACKING_INTERVAL`: 手部追踪降频系数。
*   `FACE_MESH_TASK_PATH`: 模型文件路径。

### `ConfigManager` (in `modules/camera.py`)

管理用户偏好和摄像头配置。

*   `get_camera_info(index)`: 获取指定摄像头的校准信息。
*   `save_user_prefs()`: 保存用户上次选择的摄像头索引。
