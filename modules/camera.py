
import cv2
import numpy as np
import math
import threading
import queue
import json
import os
import time

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
    def update_camera(self, device_index, fov=None, name=None, user_configured=False, resolution=None, exposure=None, api_backend=None):
        idx_str = str(device_index)
        if idx_str not in self.cameras_data:
            self.cameras_data[idx_str] = {
                "name": f"Camera {device_index}",
                "fov": 60.0,
                "user_configured": False
            }
        
        if fov is not None:
            self.cameras_data[idx_str]["fov"] = fov
        if name is not None:
            self.cameras_data[idx_str]["name"] = name
        if user_configured:
            self.cameras_data[idx_str]["user_configured"] = True
        if resolution is not None:
            self.cameras_data[idx_str]["resolution"] = resolution
        if exposure is not None:
            self.cameras_data[idx_str]["exposure"] = exposure
        if api_backend is not None:
            self.cameras_data[idx_str]["api_backend"] = api_backend
            
        self._save_json(self.cameras_config_path, self.cameras_data)

    def get_camera_info(self, device_index):
        idx_str = str(device_index)
        return self.cameras_data.get(idx_str)

    # --- 用户偏好管理 ---
    def set_last_camera(self, index):
        self.user_prefs['last_camera_index'] = index
        self._save_json(self.user_config_path, self.user_prefs)

    def get_last_camera(self):
        return self.user_prefs.get('last_camera_index')

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

# 摄像头选择逻辑
def select_camera_device(config_manager):
    """
    选择摄像头设备。
    优化逻辑：
    1. 优先尝试使用上次保存的配置直接打开，避免全扫描导致的变焦声和延迟。
    2. 仅在首次运行或手动选择时进行全扫描。
    """
    last_index = config_manager.get_last_camera()
    
    # 策略 1: 如果有上次使用的记录，直接尝试打开
    if last_index is not None:
        print(f"尝试打开上次使用的摄像头 (索引 {last_index})...")
        # 尝试快速检测该摄像头是否存在
        temp_cap = cv2.VideoCapture(last_index, cv2.CAP_DSHOW)
        if not temp_cap.isOpened():
             temp_cap = cv2.VideoCapture(last_index, cv2.CAP_ANY)
             
        if temp_cap.isOpened():
            temp_cap.release()
            print(f"成功检测到摄像头 {last_index}，直接使用。")
            
            # 获取已保存的 FOV
            saved_info = config_manager.get_camera_info(last_index)
            default_fov = saved_info['fov'] if saved_info else 60.0
            return last_index, default_fov
        else:
            print(f"无法打开上次使用的摄像头 {last_index}，回退到扫描模式。")
    
    # 策略 2: 全扫描模式 (仅在无记录或打开失败时执行)
    print("正在扫描摄像头设备...")
    available_indices = []
    for i in range(5):
        # 优先使用 DSHOW (Windows) 速度更快
        temp_cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not temp_cap.isOpened():
            temp_cap = cv2.VideoCapture(i, cv2.CAP_ANY)
        
        if temp_cap.isOpened():
            available_indices.append(i)
            config_manager.update_camera(i)
            temp_cap.release()
    
    if not available_indices:
        print("错误：未检测到摄像头设备。")
        return None, 60.0
        
    # 如果只有一个设备，直接使用
    if len(available_indices) == 1:
        selected_index = available_indices[0]
        print(f"检测到单个摄像头 (索引 {selected_index})，自动连接。")
    else:
        # 多个设备，让用户选择
        print("检测到多个摄像头设备：")
        for idx in available_indices:
            info = config_manager.get_camera_info(idx)
            fov_str = f" (FOV: {info['fov']}°)" if info else ""
            name_str = f" [{info['name']}]" if info else ""
            print(f" - 设备索引: {idx}{name_str}{fov_str}")
            
        while True:
            try:
                sel = input(f"请输入摄像头索引 {available_indices}: ").strip()
                if not sel and last_index in available_indices: # 允许回车默认
                     selected_index = last_index
                     break
                idx = int(sel)
                if idx in available_indices:
                    selected_index = idx
                    break
                print("无效的索引。")
            except ValueError:
                print("请输入数字。")
    
    # 保存选择
    config_manager.set_last_camera(selected_index)
    
    # FOV 配置逻辑
    saved_info = config_manager.get_camera_info(selected_index)
    default_fov = saved_info['fov'] if saved_info else 60.0
    
    if saved_info and saved_info.get("user_configured", False):
        print(f"使用已保存配置：FOV {default_fov}°")
        return selected_index, default_fov
    
    while True:
        try:
            val = input(f"请输入摄像头FOV [默认{default_fov}]: ").strip()
            fov_val = float(val) if val else default_fov
            if 0 < fov_val < 180:
                config_manager.update_camera(selected_index, fov=fov_val, user_configured=True)
                return selected_index, fov_val
            print("FOV 必须在 0-180 之间。")
        except ValueError:
            print("请输入数字。")

def select_resolution(cap, camera_index, config_manager):
    """
    选择分辨率。
    优化逻辑：优先使用配置文件中的分辨率，避免重新扫描导致的变焦和延迟。
    """
    # 1. 优先读取配置
    saved_info = config_manager.get_camera_info(camera_index)
    if saved_info and "resolution" in saved_info:
        w, h = saved_info["resolution"]
        print(f"使用已保存的分辨率配置: {w}x{h}")
        return w, h

    # 2. 扫描支持的分辨率 (仅在首次配置时执行)
    print("正在扫描摄像头支持的分辨率（可能会有机械变焦声）...")
    candidates = [(1920, 1080), (1280, 720), (800, 600), (640, 480)]
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
            sel = input(f"请输入编号 [0-{len(available)-1}]: ").strip()
            idx = int(sel) if sel else 0
            if 0 <= idx < len(available):
                res = available[idx]
                config_manager.update_camera(camera_index, resolution=res)
                return res
            print("无效编号。")
        except ValueError:
            print("请输入数字。")

