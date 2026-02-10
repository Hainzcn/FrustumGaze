
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
    def update_camera(self, device_index, fov=None, name=None, user_configured=False, resolution=None, exposure=None):
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
    def __init__(self, src=0, width=1920, height=1080, api_preference=cv2.CAP_ANY, queue_size=2, exposure=-5.0):
        self.src = src
        self.width = width
        self.height = height
        self.api_preference = api_preference
        self.exposure = exposure
        
        # 初始化摄像头
        self.stream = cv2.VideoCapture(self.src, self.api_preference)
        
        # 优化配置
        # 1. 强制 MJPEG 压缩
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        # 2. 设置分辨率
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # 3. 设置 FPS
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        # 4. 减少 OpenCV 内部缓冲区
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # 5. 关闭自动曝光、白平衡、增益
        try:
            self.stream.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) # 0.25 通常对应 'Manual' 模式 (具体取决于后端)
            self.stream.set(cv2.CAP_PROP_EXPOSURE, self.exposure) 
            self.stream.set(cv2.CAP_PROP_AUTO_WB, 0)
            self.stream.set(cv2.CAP_PROP_WB_TEMPERATURE, 4600)
            self.stream.set(cv2.CAP_PROP_GAIN, 32) 
            print(f"尝试设置曝光为 {self.exposure} (手动模式)...")
        except Exception as e:
            print(f"设置摄像头手动参数失败: {e}")
        
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

        # 使用 Queue 替代简单的变量 (Thread-safe)
        # maxsize 限制队列长度，防止堆积
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.frame_id = 0

    def start(self):
        if self.stopped:
            return self
        print("启动视频流读取线程...")
        self.t = threading.Thread(target=self.update, args=())
        self.t.daemon = True
        self.t.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            
            # 读取下一帧
            grabbed, frame = self.stream.read()
            
            if not grabbed:
                self.stopped = True
                return

            self.frame_id += 1
            
            # 尝试放入队列，如果满了则移除旧的（非阻塞尝试）
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            
            try:
                self.frame_queue.put((frame, self.frame_id), block=False)
            except queue.Full:
                pass # Should not happen if we just removed one, but safe guard

    def read(self):
        # 非阻塞获取最新帧
        try:
            return True, self.frame_queue.get_nowait()
        except queue.Empty:
            return False, (None, -1)
    
    def get(self, propId):
        return self.stream.get(propId)

    def stop(self):
        self.stopped = True
        if hasattr(self, 't'):
            self.t.join()
        self.stream.release()

# 摄像头选择逻辑
def select_camera_device(config_manager):
    print("正在扫描摄像头设备...")
    available_indices = []
    # 扫描常用索引 0-4
    for i in range(5):
        # 尝试快速打开检测 (优先 DSHOW)
        temp_cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not temp_cap.isOpened():
            temp_cap = cv2.VideoCapture(i, cv2.CAP_ANY)
        
        if temp_cap.isOpened():
            available_indices.append(i)
            # 自动存入/更新配置文件
            config_manager.update_camera(i)
            temp_cap.release()
    
    if not available_indices:
        print("错误：未检测到摄像头设备。")
        return None, 60.0
        
    selected_index = None
    
    # 尝试自动加载上次使用的摄像头
    last_index = config_manager.get_last_camera()
    if last_index is not None and last_index in available_indices:
        print(f"检测到上次使用的摄像头 (索引 {last_index})，按 Enter 自动选择，或输入其他索引...")
    
    if len(available_indices) == 1:
        print(f"检测到单个摄像头 (索引 {available_indices[0]})，自动连接。")
        selected_index = available_indices[0]
    else:
        print("检测到多个摄像头设备：")
        for idx in available_indices:
            info = config_manager.get_camera_info(idx)
            fov_display = ""
            name_display = ""
            if info:
                fov_display = f" (FOV: {info['fov']}°)"
                name_display = f" [{info['name']}]"
                
            marker = " *" if idx == last_index else ""
            print(f" - 设备索引: {idx}{name_display}{fov_display}{marker}")
            
        while True:
            try:
                prompt = f"请输入要使用的摄像头索引 {available_indices}"
                if last_index is not None and last_index in available_indices:
                    prompt += f" [默认 {last_index}]"
                prompt += ": "
                
                selection = input(prompt).strip()
                
                if not selection and last_index is not None and last_index in available_indices:
                    selected_index = last_index
                    break
                
                idx = int(selection)
                if idx in available_indices:
                    selected_index = idx
                    break
                else:
                    print("输入的索引无效，请重新输入。")
            except ValueError:
                print("输入格式错误，请输入数字索引。")
    
    # 保存选择
    config_manager.set_last_camera(selected_index)
    
    # 获取FOV参数
    saved_info = config_manager.get_camera_info(selected_index)
    default_fov = saved_info['fov'] if saved_info else 60.0
    
    # 检查是否已有用户配置
    if saved_info and saved_info.get("user_configured", False):
        print(f"检测到已保存的配置参数：FOV {default_fov}° (跳过输入)")
        return selected_index, default_fov
    
    while True:
        try:
            fov_input = input(f"请输入摄像头FOV(视场角) [默认{default_fov}]: ").strip()
            if not fov_input:
                fov_val = default_fov
            else:
                fov_val = float(fov_input)
            
            if 0 < fov_val < 180:
                # 用户确认输入后，更新配置文件并标记为已配置
                config_manager.update_camera(selected_index, fov=fov_val, user_configured=True)
                print(f"已保存配置：索引 {selected_index}, FOV {fov_val}°")
                return selected_index, fov_val
            else:
                print("FOV必须在0到180之间。")
        except ValueError:
            print("输入格式错误，请输入数字。")
    
    return selected_index, fov_val

def select_resolution(cap, camera_index, config_manager):
    # 1. 检查是否有保存的分辨率配置
    saved_info = config_manager.get_camera_info(camera_index)
    if saved_info and "resolution" in saved_info:
        res = saved_info["resolution"]
        print(f"检测到已保存的分辨率配置: {res[0]}x{res[1]}")
        return res[0], res[1]

    # 2. 如果没有保存配置，扫描支持的分辨率
    print("正在扫描摄像头支持的分辨率...")
    target_resolutions = [
        (1920, 1080), # 1080p
        (1280, 720),  # 720p
        (800, 600),   # SVGA
        (640, 480)    # VGA
    ]
    
    available_resolutions = []
    
    # 保存当前的设置以便恢复（可选，这里主要是为了测试）
    current_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    current_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    for w, h in target_resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        
        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 只有当实际分辨率与请求的一致时才认为是完美支持
        if aw == w and ah == h:
            available_resolutions.append((w, h))
        else:
            # 如果不完全匹配，但之前没有添加过这个实际分辨率，也可以添加
            # 但为了用户体验，只列出精准匹配的，除非一个都匹配不上
            pass
            
    # 如果没有扫描到精准匹配的，就回退到默认
    if not available_resolutions:
        print("未能检测到标准分辨率，将使用默认分辨率。")
        return int(current_w), int(current_h)
        
    print("请选择要使用的分辨率：")
    for i, (w, h) in enumerate(available_resolutions):
        print(f" {i}: {w}x{h}")
        
    selected_res = None
    while True:
        try:
            choice = input(f"请输入分辨率编号 [0-{len(available_resolutions)-1}]: ").strip()
            if not choice:
                # 默认选第一个（通常是最高的）
                selected_res = available_resolutions[0]
                break
            
            idx = int(choice)
            if 0 <= idx < len(available_resolutions):
                selected_res = available_resolutions[idx]
                break
            else:
                print("无效的编号。")
        except ValueError:
            print("请输入数字。")
            
    # 保存选择
    print(f"已选择分辨率: {selected_res[0]}x{selected_res[1]}，并保存到配置。")
    config_manager.update_camera(camera_index, resolution=selected_res)
    
    return selected_res[0], selected_res[1]
