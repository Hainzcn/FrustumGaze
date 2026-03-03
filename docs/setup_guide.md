# 安装与配置指南

## 环境要求

*   **操作系统**: Windows 10/11 (推荐), Linux (部分摄像头 API 可能需要调整)
*   **Python**: 3.8 - 3.11
*   **Unity**: 2020.3 LTS 或更高版本 (用于运行客户端 Demo)

## Python 环境搭建

1.  **克隆项目**
    ```bash
    git clone https://github.com/Hainzcn/FrustumGaze.git
    cd FrustumGaze
    ```

2.  **创建虚拟环境 (推荐)**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **安装依赖**
    ```bash
    pip install -r requirements.txt
    ```
    *如果没有 requirements.txt，请手动安装核心依赖:*
    ```bash
    pip install opencv-python mediapipe numpy
    ```

4.  **模型文件准备**
    确保 `models/` 目录下包含以下文件：
    *   `face_landmarker.task`
    *   `hand_landmarker.task`
    
    *如果缺失，请从 Google MediaPipe 官方仓库下载对应的 Task 文件。*

## 运行项目

### 1. 启动 Python 服务端

在项目根目录下运行：

```bash
python main.py
```

*   程序启动后会尝试打开默认摄像头。
*   如果成功，将弹出一个名为 "Frustum Gaze Tracking" 的窗口，显示摄像头画面及追踪骨架。
*   控制台会输出当前的 FPS 和状态信息。

### 2. 启动 Unity 客户端

1.  打开 Unity Hub，添加项目中的 `UnityProject` 文件夹（如果提供了）。
2.  打开 `SampleScene`。
3.  在场景中找到 `GameManager` 或 `NetworkManager` 物体，确保 UDP 接收端口与 Python 端配置一致 (默认 8888)。
4.  点击 Unity 编辑器的 Play 按钮。
5.  你应该能看到 Unity 中的摄像机或角色根据你的头部运动产生视差效果。

## 配置调整

### 修改摄像头

如果有多于一个摄像头，可以通过修改 `config/user_prefs.json` 或在代码中指定索引。

### 调整性能参数

编辑 `config/settings.py`：

*   **降低 CPU 占用**: 增加 `EYE_TRACKING_INTERVAL` 和 `HAND_TRACKING_INTERVAL` 的值（例如设为 2 或 3）。
*   **关闭可视化**: 将 `VISUALIZE` 设为 `False`，可显著提高 FPS，适用于生产环境。

### 网络配置

如果 Python 和 Unity 运行在不同机器上：
1.  修改 `config/settings.py` 中的 `UDP_IP` 为 Unity 运行机器的局域网 IP。
2.  确保防火墙允许 UDP 8888 端口通信。
