# 常见问题排查 (Troubleshooting)

## 1. 摄像头无法打开

**现象**: 程序启动即报错，或卡在 "Searching for camera..."。

**解决方案**:
*   检查摄像头是否被其他程序（如 Zoom, Teams）占用。
*   在 `modules/camera.py` 中，尝试更改 `cv2.VideoCapture` 的后端 API，例如强制使用 `cv2.CAP_DSHOW` (DirectShow)。
*   检查 `config/cameras.json` 中的配置是否正确。

## 2. 帧率极低 (< 10 FPS)

**现象**: 视频卡顿，延迟严重。

**解决方案**:
*   **光照**: MediaPipe 在暗光环境下性能会下降，请确保面部光照充足。
*   **CPU 瓶颈**: 检查任务管理器。如果 CPU 占用过高，尝试在 `config/settings.py` 中调高 `EYE_TRACKING_INTERVAL` (如 2 或 3)。
*   **USB 带宽**: 如果使用高分辨率摄像头，USB 带宽可能不足。尝试降低分辨率或帧率。

## 3. Unity 端无反应

**现象**: Python 端运行正常，数据显示也在发送，但 Unity 端画面静止。

**解决方案**:
*   **端口冲突**: 确保没有其他程序占用 8888 端口。
*   **防火墙**: 检查 Windows 防火墙是否拦截了 Python 的 UDP 发送或 Unity 的 UDP 接收。
*   **IP 地址**: 确认 `config/settings.py` 中的 `UDP_IP` 设置正确。如果是本机测试，应为 `127.0.0.1`。
*   **数据格式**: 检查 Python 端发送的数据结构是否与 Unity 端 C# 脚本 (`DataManager.cs`) 的解析逻辑匹配。

## 4. 追踪抖动严重

**现象**: 头部静止时，虚拟画面仍不断微颤。

**解决方案**:
*   **调整滤波参数**: 在 `config/settings.py` 中调整 OneEuroFilter 的参数。
    *   减小 `MIN_CUTOFF` (如 0.1) 可以增加平滑度，但会增加延迟。
    *   减小 `BETA` 可以减少快速运动时的抖动。
*   **检查距离**: 离摄像头太远会导致关键点识别不稳定。建议距离 50cm - 80cm。

## 5. MediaPipe 初始化失败

**现象**: 报错 `RuntimeError: Failed to load model...`

**解决方案**:
*   确认 `models/` 目录下是否存在 `.task` 文件。
*   确认 `config/settings.py` 中的路径配置是否正确指向了模型文件。
