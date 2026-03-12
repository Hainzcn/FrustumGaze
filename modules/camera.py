
import cv2
import numpy as np
import math
import threading
import queue
import json
import os
import time
import subprocess
import shutil

# --- 摄像头信息获取 ---
def get_dshow_device_map():
    """
    使用 ffmpeg 获取 DirectShow 视频设备列表，并尝试解析出 PnP ID。
    返回一个字典: {dshow_index: {'name': name, 'id': pnp_id_or_fallback}}
    """
    if not shutil.which("ffmpeg"):
        print("警告: 未找到 ffmpeg，无法自动获取准确的设备 ID 映射。")
        return {}

    cmd = ["ffmpeg", "-list_devices", "true", "-f", "dshow", "-i", "dummy"]
    try:
        # ffmpeg 输出设备列表到 stderr
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        output = result.stderr
        
        devices = {}
        current_index = 0
        lines = output.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            # 匹配设备名行: [dshow @ ...] "Device Name" (video)
            if '"' in line and '(video)' in line and not line.startswith("Error"):
                # 提取设备名
                start = line.find('"') + 1
                end = line.rfind('"')
                if start > 0 and end > start:
                    dev_name = line[start:end]
                    
                    # 尝试在下一行找 Alternative name (通常包含 PnP ID)
                    dev_id = None
                    if i + 1 < len(lines):
                        next_line = lines[i+1].strip()
                        if "Alternative name" in next_line and '"' in next_line:
                            alt_start = next_line.find('"') + 1
                            alt_end = next_line.rfind('"')
                            alt_name = next_line[alt_start:alt_end]
                            
                            # 解析 PnP ID: @device_pnp_\\?\usb#vid_xxxx&pid_xxxx...
                            # 目标格式: USB\VID_XXXX&PID_XXXX...
                            if "usb#" in alt_name.lower():
                                try:
                                    # 提取主要部分
                                    parts = alt_name.split('#')
                                    if len(parts) >= 3:
                                        # parts[1] is vid_...&pid_...
                                        # parts[2] is serial/unique id
                                        raw_id = f"USB\\{parts[1]}\\{parts[2]}"
                                        dev_id = raw_id.upper()
                                except:
                                    pass
                    
                    # 如果没解析出来，用名称做临时 ID (不太可靠但比索引好)
                    if not dev_id:
                        dev_id = dev_name
                        
                    devices[current_index] = {'name': dev_name, 'id': dev_id}
                    current_index += 1
                    
        return devices
    except Exception as e:
        print(f"获取 DirectShow 设备映射失败: {e}")
        return {}

def get_system_camera_info():
    """
    使用 PowerShell 获取系统中的摄像头设备信息。
    返回一个列表，每个元素包含 FriendlyName, InstanceId 等。
    """
    cmd = [
        "powershell",
        "-Command",
        "Get-PnpDevice | Where-Object { $_.Class -eq 'Camera' -or $_.Class -eq 'Image' } | Select-Object FriendlyName, InstanceId, Status | ConvertTo-Json"
    ]
    try:
        # 创建 STARTUPINFO 结构以隐藏控制台窗口 (Windows only)
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
        result = subprocess.run(cmd, capture_output=True, text=True, startupinfo=startupinfo)
        if result.returncode != 0:
            return []
            
        output = result.stdout.strip()
        if not output:
            return []
            
        devices = json.loads(output)
        if isinstance(devices, dict):
            devices = [devices]
            
        # 过滤状态正常的设备
        valid_devices = [d for d in devices if d.get('Status') == 'OK']
        return valid_devices
    except Exception as e:
        print(f"获取摄像头信息失败: {e}")
        return []

# --- 相机模型 ---
class CameraModel:
    def __init__(self, frame_w, frame_h, fov_deg=60.0):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.fov = fov_deg
        
        # 一次性计算内参
        self.fov_rad = math.radians(self.fov)
        # 假设像素为正方形且主点位于中心
        self.focal_length = (self.frame_w / 2) / math.tan(self.fov_rad / 2)
        self.cx = self.frame_w / 2.0
        self.cy = self.frame_h / 2.0
        
        self.cam_matrix = np.array([
            [self.focal_length, 0, self.cx],
            [0, self.focal_length, self.cy],
            [0, 0, 1]
        ], dtype="double")
        self.dist_coeffs = np.zeros((4, 1))

# --- 持久化存储管理 ---
class ConfigManager:
    def __init__(self, config_dir="config"):
        self.config_dir = config_dir
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)
            
        self.cameras_config_path = os.path.join(self.config_dir, "cameras.json")
        self.user_config_path = os.path.join(self.config_dir, "user_prefs.json")
        
        self.cameras_data = self._load_json(self.cameras_config_path)
        self.user_prefs = self._load_json(self.user_config_path)

    def _load_json(self, path):
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_json(self, path, data):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    # --- 摄像头数据管理 ---
    def update_camera(self, device_id, fov=None, name=None, user_configured=False, resolution=None, exposure=None, api_backend=None, last_index=None):
        """
        更新摄像头配置。
        device_id: 摄像头的唯一标识符 (InstanceId)
        """
        if device_id not in self.cameras_data:
            self.cameras_data[device_id] = {
                "name": name if name else f"Camera {device_id}",
                "fov": 60.0,
                "user_configured": False
            }
        
        if fov is not None:
            self.cameras_data[device_id]["fov"] = fov
        if name is not None:
            self.cameras_data[device_id]["name"] = name
        if user_configured:
            self.cameras_data[device_id]["user_configured"] = True
        if resolution is not None:
            self.cameras_data[device_id]["resolution"] = resolution
        if exposure is not None:
            self.cameras_data[device_id]["exposure"] = exposure
        if api_backend is not None:
            self.cameras_data[device_id]["api_backend"] = api_backend
        if last_index is not None:
            self.cameras_data[device_id]["last_index"] = last_index
            
        self._save_json(self.cameras_config_path, self.cameras_data)

    def get_camera_info(self, device_id):
        return self.cameras_data.get(device_id)

    # --- 用户偏好管理 ---
    def set_last_camera(self, device_id):
        self.user_prefs['last_camera_id'] = device_id
        self._save_json(self.user_config_path, self.user_prefs)

    def get_last_camera(self):
        # 优先读取新的 ID 格式
        last_id = self.user_prefs.get('last_camera_id')
        if last_id:
            return last_id
        # 兼容旧版索引
        return str(self.user_prefs.get('last_camera_index')) if 'last_camera_index' in self.user_prefs else None

    def should_scan_new_cameras(self):
        """检查是否需要扫描新摄像头，默认为 False (除非首次运行或被重置)"""
        # 如果没有配置过，说明是首次运行，应当扫描
        if 'check_new_camera' not in self.user_prefs:
            return True
        return self.user_prefs.get('check_new_camera', False)

    def set_scan_new_cameras(self, should_scan):
        self.user_prefs['check_new_camera'] = should_scan
        self._save_json(self.user_config_path, self.user_prefs)


# --- 视频流获取优化 (Producer) ---
class WebcamVideoStream:
    """
    视频流捕获类。
    负责从摄像头读取帧，支持多线程读取以提高性能，
    并支持写入共享内存以供多进程使用。
    """
    def __init__(self, src=0, width=1920, height=1080, api_preference=cv2.CAP_ANY, queue_size=2, exposure=-5.0, shm_arrays=None):
        self.src = src
        self.width = width
        self.height = height
        self.api_preference = api_preference
        self.exposure = exposure
        self.shm_arrays = shm_arrays # 共享内存数组列表
        
        # 初始化摄像头
        self.stream = cv2.VideoCapture(self.src, self.api_preference)
        
        # 优化配置
        # 1. 强制 MJPEG 压缩 (提高帧率)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        # 2. 设置分辨率
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # 3. 设置 FPS
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        # 4. 减少 OpenCV 内部缓冲区 (降低延迟)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # 5. 关闭自动曝光、白平衡、增益 (提高稳定性)
        try:
            self.stream.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) # 0.25 通常对应 'Manual' 模式
            self.stream.set(cv2.CAP_PROP_EXPOSURE, self.exposure) 
            self.stream.set(cv2.CAP_PROP_AUTO_WB, 0)
            self.stream.set(cv2.CAP_PROP_WB_TEMPERATURE, 4600)
            self.stream.set(cv2.CAP_PROP_GAIN, 32) 
        except Exception as e:
            print(f"WebcamVideoStream: 设置摄像头手动参数失败: {e}")
        
        # 检查是否成功打开
        if not self.stream.isOpened():
            print("WebcamVideoStream: 无法打开摄像头")
            self.stopped = True
        else:
            self.stopped = False
            
        # 读取第一帧确认
        (self.grabbed, self.frame) = self.stream.read()
        if not self.grabbed:
            print("WebcamVideoStream: 无法读取第一帧")
            self.stopped = True

        # 使用队列进行线程间通信
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.frame_id = 0

    def start(self):
        """启动视频读取线程"""
        if self.stopped:
            return self
        print("启动视频流读取线程...")
        self.t = threading.Thread(target=self.update, args=())
        self.t.daemon = True
        self.t.start()
        return self

    def set_shared_memory(self, shm_arrays):
        """延迟设置共享内存 (支持双缓冲列表)"""
        if isinstance(shm_arrays, list):
            self.shm_arrays = shm_arrays
        else:
            self.shm_arrays = [shm_arrays]

    def update(self):
        """后台线程持续读取帧"""
        while True:
            if self.stopped:
                return
            
            grabbed, frame = self.stream.read()
            
            if not grabbed:
                self.stopped = True
                return

            self.frame_id += 1
            if self.frame_id > 1000000000:
                self.frame_id = 0
            
            # 如果配置了共享内存，直接写入
            buffer_idx = -1
            if self.shm_arrays is not None and frame is not None:
                try:
                    # 双缓冲逻辑: 根据 frame_id 奇偶性选择 buffer
                    buffer_idx = self.frame_id % len(self.shm_arrays)
                    target_shm = self.shm_arrays[buffer_idx]
                    
                    if frame.shape == target_shm.shape:
                        np.copyto(target_shm, frame)
                except Exception:
                    pass

            # 尝试放入队列 (非阻塞)
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            
            try:
                # 如果使用共享内存，队列中只放元数据以减少拷贝
                if self.shm_arrays is not None:
                    self.frame_queue.put((None, self.frame_id, buffer_idx), block=False)
                else:
                    self.frame_queue.put((frame, self.frame_id, -1), block=False)
            except queue.Full:
                pass

    def read(self):
        """获取最新帧"""
        try:
            return True, self.frame_queue.get_nowait()
        except queue.Empty:
            return False, (None, -1, -1)
    
    def get(self, propId):
        return self.stream.get(propId)

    def stop(self):
        """停止读取并释放资源"""
        self.stopped = True
        if hasattr(self, 't'):
            self.t.join()
        self.stream.release()

def select_camera_device(config_manager):
    """
    选择摄像头设备。
    逻辑优化：
    1. 使用 ffmpeg 获取 DirectShow 设备列表及其 ID (与 OpenCV 索引一致)。
    2. 如果有 'last_camera_id' 且在列表中找到匹配项，直接返回。
    3. 如果没有记录或无法匹配，进行扫描并列出设备供用户选择。
    4. 自动关联 ID，无需用户手动确认物理设备。
    """
    last_id = config_manager.get_last_camera()
    should_scan = config_manager.should_scan_new_cameras()
    
    # 获取 DirectShow 设备映射 (Index -> ID)
    dshow_map = get_dshow_device_map()
    
    # 策略 1: 快速启动 (基于 ID 匹配)
    if not should_scan and last_id:
        target_index = None
        
        # 遍历 dshow_map 查找 ID 匹配的索引
        for idx, info in dshow_map.items():
            # 尝试完全匹配 ID
            if info['id'] == last_id:
                target_index = idx
                print(f"快速启动：匹配到设备 '{info['name']}' (索引 {idx})")
                break
            
            # 尝试部分匹配 (例如 last_id 是具体的 InstanceId，而 dshow 提取的是 generic ID)
            # 或者反过来
            if last_id in info['id'] or info['id'] in last_id:
                 target_index = idx
                 print(f"快速启动：模糊匹配到设备 '{info['name']}' (索引 {idx})")
                 break

        if target_index is not None:
             # 验证可用性
             temp_cap = cv2.VideoCapture(target_index, cv2.CAP_DSHOW)
             if not temp_cap.isOpened():
                 temp_cap = cv2.VideoCapture(target_index, cv2.CAP_ANY)
                 
             if temp_cap.isOpened():
                 temp_cap.release()
                 # 更新 last_index 以备后用
                 config_manager.update_camera(last_id, last_index=target_index)
                 
                 saved_info = config_manager.get_camera_info(last_id)
                 default_fov = saved_info['fov'] if saved_info else 60.0
                 return target_index, default_fov
             else:
                 print("上次使用的设备无法打开，转入扫描模式。")
        else:
             print("无法定位上次使用的设备 ID，转入扫描模式。")

    # 策略 2: 扫描模式
    print("正在扫描摄像头设备...")
    
    # 如果 dshow_map 为空 (ffmpeg 失败)，回退到简单的索引扫描
    available_indices = []
    
    if dshow_map:
        print("检测到以下 DirectShow 设备:")
        for idx, info in dshow_map.items():
            # 验证是否真能打开
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                available_indices.append(idx)
                cap.release()
                print(f" [{idx}] {info['name']} (ID: {info['id']})")
            else:
                print(f" [x] {info['name']} (无法打开)")
    else:
        # 回退逻辑
        print("警告: 无法获取设备名称映射，仅显示可用索引。")
        for i in range(6):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if not cap.isOpened(): cap = cv2.VideoCapture(i, cv2.CAP_ANY)
            if cap.isOpened():
                available_indices.append(i)
                cap.release()
                print(f" [{i}] Camera {i}")

    if not available_indices:
        print("错误：未检测到任何可用摄像头。")
        return None, 60.0

    # 用户选择
    selected_index = None
    if len(available_indices) == 1:
        selected_index = available_indices[0]
        print(f"自动选择唯一可用设备 (索引 {selected_index})。")
    else:
        while True:
            try:
                sel = input(f"请输入要使用的摄像头索引 {available_indices}: ").strip()
                idx = int(sel)
                if idx in available_indices:
                    selected_index = idx
                    break
                print("无效的索引。")
            except ValueError:
                print("请输入数字。")

    # 自动保存配置 (使用 dshow_map 中的 ID)
    selected_dev_id = str(selected_index) # 默认回退
    selected_dev_name = f"Camera {selected_index}"
    
    if dshow_map and selected_index in dshow_map:
        info = dshow_map[selected_index]
        selected_dev_id = info['id']
        selected_dev_name = info['name']
        print(f"已关联设备 ID: {selected_dev_id}")
    
    # 更新配置
    config_manager.update_camera(
        selected_dev_id, 
        name=selected_dev_name, 
        last_index=selected_index
    )
    config_manager.set_last_camera(selected_dev_id)
    
    # 扫描完成标志重置
    if should_scan:
        config_manager.set_scan_new_cameras(False)

    # FOV 配置
    saved_info = config_manager.get_camera_info(selected_dev_id)
    default_fov = saved_info.get('fov', 60.0)
    
    if saved_info and saved_info.get("user_configured", False):
        print(f"加载已保存配置：FOV {default_fov}°")
        return selected_index, default_fov
    
    while True:
        try:
            val = input(f"请输入摄像头FOV [默认{default_fov}]: ").strip()
            fov_val = float(val) if val else default_fov
            if 0 < fov_val < 180:
                config_manager.update_camera(selected_dev_id, fov=fov_val, user_configured=True)
                return selected_index, fov_val
            print("FOV 必须在 0-180 之间。")
        except ValueError:
            print("请输入数字。")

def select_resolution(cap, camera_index, config_manager):
    """
    选择分辨率。
    优化逻辑：优先使用配置文件中的分辨率，避免重新扫描导致的变焦和延迟。
    """
    # 尝试获取当前选中的设备 ID (由 select_camera_device 设置)
    device_id = config_manager.get_last_camera()
    
    # 如果未找到 ID (例如用户跳过关联)，则回退到使用索引作为临时 ID
    if not device_id:
        device_id = str(camera_index)

    # 1. 优先读取配置
    saved_info = config_manager.get_camera_info(device_id)
    if saved_info and "resolution" in saved_info:
        w, h = saved_info["resolution"]
        print(f"使用已保存的分辨率配置: {w}x{h}")
        return w, h

    # 2. 扫描支持的分辨率 (仅在首次配置时执行)
    print("正在扫描摄像头支持的分辨率（可能会有机械变焦声）...")
    # 从高到低尝试
    candidates = [
        (3840, 2160), # 4K
        (2560, 1440), # 2K
        (1920, 1080), # 1080p
        (1280, 720),  # 720p
        (800, 600),   # SVGA
        (640, 480)    # VGA
    ]
    available = []
    
    current_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    current_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    for w, h in candidates:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        if int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == w and int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == h:
            available.append((w, h))
            
    # 恢复默认分辨率以免影响后续
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, current_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, current_h)

    if not available:
        print("未能检测到标准分辨率，将使用默认值。")
        return int(current_w), int(current_h)
        
    print("请选择分辨率：")
    for i, (w, h) in enumerate(available):
        print(f" {i}: {w}x{h}")
        
    while True:
        try:
            sel = input(f"请输入编号 [0-{len(available)-1}] (默认0): ").strip()
            idx = int(sel) if sel else 0
            if 0 <= idx < len(available):
                res = available[idx]
                config_manager.update_camera(device_id, resolution=res)
                return res
            print("无效编号。")
        except ValueError:
            print("请输入数字。")


