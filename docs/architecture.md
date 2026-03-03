# 系统架构文档

## 1. 总体架构

FrustumGaze 采用多进程架构设计，旨在解决 Python 在处理高分辨率视频流和复杂计算机视觉任务（MediaPipe）时的性能瓶颈（GIL）。

核心设计理念：
*   **主进程 (Main Process)**: 负责视频采集、渲染显示、任务分发、数据聚合与网络发送。
*   **子进程 (Worker Processes)**: 负责耗时的计算机视觉计算（人脸追踪、手部追踪）。
*   **共享内存 (Shared Memory)**: 用于在进程间高效传输图像数据，避免序列化/反序列化的开销。
*   **队列 (Queues)**: 用于轻量级的任务指令和结果传递。

### 架构图

```mermaid
graph TD
    Camera[摄像头输入] -->|Capture| Main[主进程 (Pipeline)]
    Main -->|写入| SharedMem[共享内存 (Double Buffer)]
    
    subgraph "主进程 (Main)"
        Main -->|Visualizer| Display[屏幕显示]
        Main -->|UDPSender| Unity[Unity 客户端]
    end
    
    subgraph "人脸追踪子进程 (Face Process)"
        SharedMem -->|读取| FaceTask[FaceLandmarker]
        FaceTask -->|结果| FaceQueue[结果队列]
    end
    
    subgraph "手部追踪子进程 (Hand Process)"
        SharedMem -->|读取| HandTask[HandLandmarker]
        HandTask -->|结果| HandQueue[结果队列]
    end
    
    Main -->|Task Queue| FaceTask
    Main -->|Task Queue| HandTask
    FaceQueue -->|Result| Main
    HandQueue -->|Result| Main
```

## 2. 核心模块详解

### 2.1 Pipeline (`modules/pipeline.py`)

`FrustumGazePipeline` 是系统的核心控制器，负责生命周期管理。

*   **初始化**: 启动摄像头，申请共享内存，创建并启动子进程。
*   **主循环 (`run`)**:
    1.  从摄像头读取一帧。
    2.  将帧数据写入共享内存（使用双缓冲策略避免读写冲突）。
    3.  根据设定的频率（`EYE_TRACKING_INTERVAL`, `HAND_TRACKING_INTERVAL`）向子进程发送任务指令。
    4.  非阻塞地尝试从子进程的结果队列中获取最新的追踪结果。
    5.  利用最新的追踪结果进行视线解算 (PnP Solver) 和数据平滑 (Kalman/OneEuro Filter)。
    6.  绘制可视化信息。
    7.  打包数据并通过 UDP 发送给 Unity。
    8.  统计 FPS 和性能指标。

### 2.2 共享内存机制 (`modules/shared_mem.py`)

为了在进程间高效传输 1080p+ 的图像数据，系统使用了 `multiprocessing.shared_memory`。

*   **双缓冲 (Double Buffering)**: 创建两个共享内存块。主进程写入 Buffer A 时，子进程可能正在读取 Buffer B，反之亦然。虽然目前实现主要依赖 Python 的 GIL 和队列同步来避免冲突，但设计上支持缓冲切换。
*   **Zero-Copy**: 子进程直接通过 `numpy.ndarray` 映射访问共享内存，无需数据拷贝。

### 2.3 追踪子进程 (`trackers/`)

*   **FaceProcess (`trackers/face_mesh.py`)**:
    *   加载 `face_landmarker.task` 模型。
    *   接收任务指令（包含 Frame ID 和 Buffer Index）。
    *   从共享内存读取图像。
    *   运行 MediaPipe Face Mesh。
    *   提取关键点（虹膜、眼睑等）并返回。
    *   **ROI 优化**: 如果上一帧检测成功，下一帧会根据上一帧的人脸位置裁剪出一个 ROI (Region of Interest) 进行检测，大大减少计算量。如果丢失追踪，则自动切换回全图扫描。

*   **HandProcess (`trackers/hand_tracker.py`)**:
    *   加载 `hand_landmarker.task` 模型。
    *   逻辑与 FaceProcess 类似，独立运行，互不干扰。

### 2.4 数据通信 (`modules/network.py` & Unity)

*   **协议**: UDP
*   **数据格式**: 自定义二进制或 JSON（视实现而定，当前版本主要发送结构化数据）。
*   **内容**:
    *   头部姿态 (Head Pose: Translation, Rotation)
    *   注视点数据 (Gaze Vector)
    *   手部骨骼关键点
    *   交互事件（如捏合手势）

## 3. 坐标系转换

系统涉及多个坐标系的转换：

1.  **图像坐标系 (2D)**: 像素单位 (0,0) 到 (Width, Height)。
2.  **相机坐标系 (3D)**: OpenCV 标准，X 右，Y 下，Z 前。
3.  **模型坐标系 (3D)**: 用于 solvePnP 的标准化人脸 3D 模型。
4.  **Unity 坐标系 (3D)**: 左手坐标系，Y 上，Z 前。

在发送数据前，Python 端或接收端的 C# 脚本需要进行坐标系变换以适配 Unity 的世界坐标。

## 4. 性能优化策略

1.  **多进程**: 将最耗时的 CPU 密集型任务剥离。
2.  **动态 ROI**: 仅在人脸区域进行检测，显著提升 MediaPipe 推理速度。
3.  **跳帧策略**: 允许配置追踪频率（如每 2 帧追踪一次），中间帧使用插值或卡尔曼滤波预测。
4.  **滤波器**: 使用 OneEuroFilter 和 KalmanFilter 消除抖动，平滑数据，弥补跳帧带来的不连贯。
