import cv2
import numpy as np
import time
import queue
import multiprocessing
import sys
from collections import deque

from config.settings import VISUALIZE, UDP_IP, UDP_PORT, EYE_TRACKING_INTERVAL, HAND_TRACKING_INTERVAL, POSE_TRACKING_INTERVAL
from modules.camera import CameraModel, ConfigManager, WebcamVideoStream, select_camera_device, select_resolution
from modules.network import UDPSender
from modules.visualizer import Visualizer
from modules.shared_mem import create_shared_array
from modules.stats import StatsManager
from utils.image_utils import ImagePreprocessor
from trackers.eye_tracker import EyeTracker
from .face_process import FrameProcessorProcess
from .hand_process import HandProcessorProcess
from .pose_process import PoseProcessorProcess

"""
FrustumGaze 核心管道管理器。

负责：
- 初始化和管理摄像头视频流。
- 创建和协调多个子进程（人脸、手部、姿态追踪）。
- 处理进程间通信（共享内存和队列）。
- 收集和分发追踪结果。
- 管理性能统计和可视化渲染。
- 通过 UDP 发送追踪数据到 Unity 端。
"""

class FrustumGazePipeline:
    """
    FrustumGaze 应用程序的主管道类。
    管理整个系统的生命周期，包括摄像头设置、多进程追踪、数据传输和可视化。
    """
    def __init__(self):
        # 启用 multiprocessing 支持 (Windows 下必须)
        multiprocessing.freeze_support()
        
        # 进程间通信队列
        self.input_queue = multiprocessing.Queue(maxsize=2)  # 面部追踪输入队列
        self.output_queue = multiprocessing.Queue(maxsize=2) # 面部追踪输出队列
        self.hand_input_queue = multiprocessing.Queue(maxsize=2) # 手部追踪输入队列
        self.hand_output_queue = multiprocessing.Queue(maxsize=2) # 手部追踪输出队列
        self.pose_input_queue = multiprocessing.Queue(maxsize=2) # 姿态追踪输入队列
        self.pose_output_queue = multiprocessing.Queue(maxsize=2) # 姿态追踪输出队列
        self.stop_event = multiprocessing.Event() # 用于通知子进程停止的事件
        
        # 核心功能管理器
        self.config_manager = ConfigManager() # 配置管理器
        self.stats_manager = StatsManager() # 性能统计管理器
        self.udp_sender = UDPSender(UDP_IP, UDP_PORT) # UDP 数据发送器
        self.visualizer = Visualizer() # 可视化工具
        
        # 追踪器实例
        self.eye_tracker = EyeTracker() # 眼动追踪器
        self.preprocessor = ImagePreprocessor() # 图像预处理器
        
        # 摄像头相关属性
        self.camera_index = None # 摄像头索引
        self.camera_fov = 60.0 # 摄像头水平视场角 (FOV)
        self.video_stream = None # 视频流对象
        self.camera_model = None # 摄像头模型（包含内参等）
        
        # 共享内存管理
        self.shm_names = [] # 共享内存名称列表
        self.shm_managers = [] # 共享内存管理器列表
        self.shm_arrays = [] # 共享 NumPy 数组列表
        self.frame_shape = None # 视频帧的形状 (高, 宽, 通道数)
        
        # 子进程实例
        self.face_process = None # 人脸处理子进程
        self.hand_process = None # 手部处理子进程
        self.pose_process = None # 姿态处理子进程
        
        # 管道运行状态
        self.running = False # 管道是否正在运行
        self.current_display_frame = None # 当前用于显示的帧
        
        # 最新检测结果缓存
        self.latest_hand_result = None # 最新手部检测原始结果
        self.latest_hands_pos = None # 最新手部位置信息
        self.latest_closest_hand = None # 最新最近手部信息
        self.latest_face_result = None # 最新人脸检测原始结果
        self.latest_pose_result = None # 最新姿态检测原始结果
        self.latest_roi_info = None # 最新感兴趣区域信息
        self.latest_using_full_scan = False # 是否正在进行全帧扫描
        self.latest_eye_points = [] # 最新眼部关键点（滤波后）
        self.latest_raw_eye_points = [] # 最新眼部关键点（原始）
        self.latest_gaze_data = None # 最新视线数据
        
        # 帧处理计数器（用于控制处理频率）
        self.hand_frame_counter = 0 # 手部追踪帧计数器
        self.pose_frame_counter = 0 # 姿态追踪帧计数器
        self.eye_frame_counter = 0 # 眼动追踪帧计数器
        
        # 视线数据容器 (用于复用，避免频繁创建对象)
        self.gaze_data_container = {
            'rvec': None, # 旋转向量
            'tvec': None, # 平移向量
            'cam_matrix': None, # 摄像头内参矩阵
            'dist_coeffs': None, # 摄像头畸变系数
            'rmat': None # 旋转矩阵
        }

    def setup(self):
        """
        初始化摄像头并设置分辨率。
        包括：
        1. 选择摄像头设备和视场角。
        2. 尝试不同的 OpenCV 后端 API 以确保兼容性。
        3. 设置摄像头分辨率和曝光值。
        4. 启动视频流并创建共享内存。
        5. 初始化相机模型。
        """
        # 1. 摄像头选择
        self.camera_index, self.camera_fov = select_camera_device(self.config_manager)
        if self.camera_index is None:
            print("未选择摄像头，退出。")
            return False

        # 2. 尝试不同的 OpenCV 后端 API
        cap_temp = None
        used_api = cv2.CAP_ANY
        api_candidates = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        
        camera_info = self.config_manager.get_camera_info(self.camera_index)
        if camera_info and "api_backend" in camera_info:
            saved_api = int(camera_info["api_backend"])
            print(f"检测到上次成功使用的 API: {saved_api}，将优先尝试。")
            if saved_api in api_candidates:
                api_candidates.remove(saved_api)
            api_candidates.insert(0, saved_api)

        for api in api_candidates:
            print(f"尝试 API: {api} ...")
            cap_temp = cv2.VideoCapture(self.camera_index, api)
            if cap_temp.isOpened():
                print(f"成功使用 API: {api}")
                used_api = api
                self.config_manager.update_camera(self.camera_index, api_backend=used_api)
                break
            else:
                print(f"API {api} 初始化失败。")

        if not cap_temp or not cap_temp.isOpened():
            print(f"错误: 无法打开摄像头 {self.camera_index}")
            return False

        # 3. 设置摄像头分辨率
        target_w, target_h = select_resolution(cap_temp, self.camera_index, self.config_manager)

        # 4. 获取并设置曝光配置
        exposure_val = -5.0 # 默认曝光值
        if camera_info and "exposure" in camera_info:
            exposure_val = float(camera_info["exposure"])
            print(f"检测到已保存的曝光配置: {exposure_val}")
        else:
            print(f"使用默认曝光值: {exposure_val}")
            self.config_manager.update_camera(self.camera_index, exposure=exposure_val)

        cap_temp.release() # 释放临时捕获对象

        # 5. 启动优化视频流 (独立线程)
        print(f"正在启动优化视频流 (MJPEG, 独立线程)...")
        print(f"目标分辨率: {target_w}x{target_h}")
        
        self.video_stream = WebcamVideoStream(
            src=self.camera_index, 
            width=target_w, 
            height=target_h, 
            api_preference=used_api, 
            exposure=exposure_val
        ).start()

        # 等待摄像头预热，确保帧数据稳定
        time.sleep(1.0)

        # 6. 读取实际分辨率并创建共享内存
        actual_w = self.video_stream.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"摄像头最终实际分辨率: {int(actual_w)}x{int(actual_h)}")

        if int(actual_w) != target_w or int(actual_h) != target_h:
            print(f"警告: 实际分辨率 ({int(actual_w)}x{int(actual_h)}) 与请求分辨率 ({target_w}x{target_h}) 不一致。")

        self.frame_shape = (int(actual_h), int(actual_w), 3)
        
        # 初始化共享内存块 (双缓冲机制，用于主进程与子进程间高效传输帧数据)
        for i in range(2):
            name = f"frustum_gaze_frame_buffer_{i}"
            try:
                mgr, arr = create_shared_array(self.frame_shape, dtype=np.uint8, name=name)
                self.shm_names.append(name)
                self.shm_managers.append(mgr)
                self.shm_arrays.append(arr)
            except Exception as e:
                print(f"创建共享内存 {name} 失败: {e}")
                return False
        
        self.video_stream.set_shared_memory(self.shm_arrays)

        # 7. 相机模型初始化 (用于姿态和深度计算)
        self.camera_model = CameraModel(actual_w, actual_h, self.camera_fov)
        self.gaze_data_container['cam_matrix'] = self.camera_model.cam_matrix
        self.gaze_data_container['dist_coeffs'] = self.camera_model.dist_coeffs

        return True

    def start_processes(self):
        """
        启动人脸、手部和姿态追踪子进程。
        每个子进程负责独立的计算任务，并通过队列和共享内存与主进程通信。
        """
        self.face_process = FrameProcessorProcess(
            self.input_queue, # 面部追踪输入队列
            self.output_queue, # 面部追踪输出队列
            self.preprocessor, # 图像预处理器
            self.stop_event, # 停止事件
            self.shm_names, # 共享内存名称
            self.frame_shape, # 帧形状
            camera_fov=self.camera_fov # 摄像头视场角
        )
        self.face_process.start()

        self.hand_process = HandProcessorProcess(
            self.hand_input_queue, # 手部追踪输入队列
            self.hand_output_queue, # 手部追踪输出队列
            self.stop_event, # 停止事件
            self.shm_names, # 共享内存名称
            self.frame_shape, # 帧形状
            fov=self.camera_fov # 摄像头视场角
        )
        self.hand_process.start()
        
        self.pose_process = PoseProcessorProcess(
            self.pose_input_queue, # 姿态追踪输入队列
            self.pose_output_queue, # 姿态追踪输出队列
            self.stop_event, # 停止事件
            self.shm_names, # 共享内存名称
            self.frame_shape # 帧形状
        )
        self.pose_process.start()

        print("管道启动: 捕获 (线程) -> 共享内存 -> 处理 (进程) -> 主循环")

    def stop(self):
        """
        停止所有子进程、视频流，并释放共享内存和 OpenCV 窗口。
        确保所有资源被正确清理。
        """
        print("正在停止所有进程...")
        self.running = False
        self.stop_event.set() # 设置停止事件，通知所有子进程退出
        
        # 清空所有队列，防止子进程因队列满而阻塞
        self._drain_queues()
        
        # 等待子进程结束，若超时则强制终止
        if self.face_process:
            self.face_process.join(timeout=2.0)
            if self.face_process.is_alive():
                print("人脸处理进程未能正常停止，尝试终止。")
                self.face_process.terminate()

        if self.hand_process:
            self.hand_process.join(timeout=2.0)
            if self.hand_process.is_alive():
                print("手部处理进程未能正常停止，尝试终止。")
                self.hand_process.terminate()

        if self.pose_process:
            self.pose_process.join(timeout=2.0)
            if self.pose_process.is_alive():
                print("姿态处理进程未能正常停止，尝试终止。")
                self.pose_process.terminate()
            
        if self.video_stream:
            self.video_stream.stop() # 停止视频流
            
        self.udp_sender.close() # 关闭 UDP 发送器
        
        # 清理共享内存
        for mgr in self.shm_managers:
            try:
                mgr.close()
                mgr.unlink()
            except Exception as e:
                print(f"清理共享内存失败: {e}")
        cv2.destroyAllWindows() # 关闭所有 OpenCV 窗口

    def _drain_queues(self):
        """清空所有队列，防止子进程因队列满而阻塞无法退出。"""
        queues = [
            self.input_queue, self.output_queue,
            self.hand_input_queue, self.hand_output_queue,
            self.pose_input_queue, self.pose_output_queue
        ]
        for q in queues:
            try:
                while not q.empty():
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break
            except Exception:
                pass

    def run(self):
        """
        启动主处理循环。
        - 调用 setup() 初始化摄像头和资源。
        - 调用 start_processes() 启动子进程。
        - 进入循环，持续处理视频帧、检查子进程结果、更新统计信息和渲染可视化。
        - 捕获 KeyboardInterrupt 信号以优雅地停止管道。
        """
        if not self.setup():
            print("管道初始化失败，退出。")
            return

        self.start_processes()
        self.running = True
        
        try:
            while self.running:
                self._process_frame() # 处理当前帧
                self._check_hand_results() # 检查手部追踪结果
                self._check_pose_results() # 检查姿态追踪结果
                self._check_face_results() # 检查面部追踪结果
                self._update_stats() # 更新性能统计
                
                if self._render(): # 渲染可视化并检查是否需要停止
                    break
                
        except KeyboardInterrupt:
            print("用户中断操作，正在停止管道。")
        finally:
            self.stop() # 确保在退出前释放所有资源

    def _process_frame(self):
        """
        从视频流获取最新帧，并根据预设频率分发给不同的追踪子进程。
        - 帧数据通过共享内存传递，避免数据复制开销。
        - 使用帧计数器控制面部、手部和姿态追踪的频率。
        """
        has_frame, frame_data = self.video_stream.read()
        if has_frame:
            frame, frame_id, buffer_idx = frame_data
            
            # 如果 frame 为 None，表示视频流已将帧写入共享内存；否则，将帧复制到共享内存
            if frame is None and buffer_idx >= 0:
                self.current_display_frame = self.shm_arrays[buffer_idx].copy()
            elif frame is not None:
                np.copyto(self.shm_arrays[0], frame)
                self.current_display_frame = frame
                buffer_idx = 0
            
            self.stats_manager.record_captured() # 记录捕获帧数
            
            # 根据 EYE_TRACKING_INTERVAL 分发面部追踪任务
            self.eye_frame_counter = (self.eye_frame_counter + 1) % EYE_TRACKING_INTERVAL
            if self.eye_frame_counter == 0:
                self.stats_manager.record_face_task_attempted()
                try:
                    self.input_queue.put({'frame_id': frame_id, 'buffer_idx': buffer_idx}, block=False)
                except queue.Full:
                    self.stats_manager.record_face_task_dropped() # 队列满时记录丢弃

            # 根据 HAND_TRACKING_INTERVAL 分发手部追踪任务
            self.hand_frame_counter = (self.hand_frame_counter + 1) % HAND_TRACKING_INTERVAL
            if self.hand_frame_counter == 0:
                self.stats_manager.record_hand_task_attempted()
                try:
                    self.hand_input_queue.put({'frame_id': frame_id, 'buffer_idx': buffer_idx}, block=False)
                except queue.Full:
                    self.stats_manager.record_hand_task_dropped() # 队列满时记录丢弃

            # 根据 POSE_TRACKING_INTERVAL 分发姿态追踪任务
            self.pose_frame_counter = (self.pose_frame_counter + 1) % POSE_TRACKING_INTERVAL
            if self.pose_frame_counter == 0:
                try:
                    self.pose_input_queue.put({'frame_id': frame_id, 'buffer_idx': buffer_idx}, block=False)
                except queue.Full:
                    pass # 队列满时忽略，不记录丢弃（姿态追踪优先级相对较低）

    def _check_hand_results(self):
        """
        检查手部追踪子进程的输出队列，获取最新的手部检测结果。
        如果检测到手部，则将最近手部的姿态和捏合状态通过 UDP 发送。
        """
        try:
            hand_result_data = self.hand_output_queue.get_nowait()
            self.latest_hand_result = hand_result_data.get('hand_result') # 原始手部检测结果
            self.latest_hands_pos = hand_result_data.get('hands_pos') # 所有手部的位置信息
            self.latest_closest_hand = hand_result_data.get('closest_hand') # 最近手部的信息
            
            # 如果有最近的手部数据，则通过 UDP 发送
            if self.latest_closest_hand:
                is_pinching = 1 if self.latest_closest_hand.get('is_pinching', False) else 0 # 捏合状态
                hx = self.latest_closest_hand.get('x', 0.0) # 手部 X 坐标
                hy = self.latest_closest_hand.get('y', 0.0) # 手部 Y 坐标
                hz = self.latest_closest_hand.get('z', 0.0) # 手部 Z 坐标
                
                hand_str = f"H:{is_pinching},{hx:.3f},{hy:.3f},{hz:.3f}"
                self.udp_sender.send(hand_str)
        except queue.Empty:
            pass # 队列为空，无新结果

    def _check_pose_results(self):
        """
        检查姿态追踪子进程的输出队列，获取最新的姿态检测结果。
        """
        try:
            pose_result_data = self.pose_output_queue.get_nowait()
            self.latest_pose_result = pose_result_data.get('pose_result') # 姿态检测原始结果
        except queue.Empty:
            pass # 队列为空，无新结果

    def _check_face_results(self):
        """
        检查面部追踪子进程的输出队列，获取最新的人脸检测和视线追踪结果。
        - 更新 EyeTracker 的内部状态。
        - 根据头部姿态（Yaw, Pitch, Roll）重建旋转矩阵。
        - 将视线数据通过 UDP 发送至 Unity。
        """
        try:
            result_data = self.output_queue.get_nowait()
            
            self.latest_face_result = result_data['detection_result'] # 原始人脸检测结果
            self.latest_roi_info = result_data['roi_info'] # 当前追踪的感兴趣区域 (ROI)
            self.latest_using_full_scan = result_data.get('using_full_scan', False) # 是否处于全帧扫描模式
            
            self.stats_manager.record_processed() # 记录处理帧
            self.stats_manager.update_fps() # 更新 FPS 统计
            
            # 初始化/重置视线显示数据
            self.latest_eye_points = []
            self.latest_raw_eye_points = []
            self.latest_gaze_data = None
            
            if not self.latest_using_full_scan:
                processed_gaze_data = result_data.get('processed_gaze_data')
                
                if processed_gaze_data:
                    # 从子进程结果更新主进程的追踪器状态
                    est_dist, off_x, off_y = processed_gaze_data['gaze_params']
                    self.eye_tracker.current_estimated_dist = est_dist
                    self.eye_tracker.current_offset_x = off_x
                    self.eye_tracker.current_offset_y = off_y
                    self.eye_tracker.current_pixel_dist = processed_gaze_data.get('current_pixel_dist', 0)
                    self.eye_tracker.head_center_pos = processed_gaze_data.get('head_center_pos')
                    self.eye_tracker.current_yaw = processed_gaze_data.get('yaw', 0.0)
                    self.eye_tracker.current_pitch = processed_gaze_data.get('pitch', 0.0)
                    self.eye_tracker.current_geo_yaw = processed_gaze_data.get('geo_yaw', 0.0)
                    self.eye_tracker.current_geo_pitch = processed_gaze_data.get('geo_pitch', 0.0)
                    self.eye_tracker.current_depth_details = processed_gaze_data.get('current_depth_details', {})

                    self.latest_eye_points = processed_gaze_data.get('eye_points', [])
                    self.latest_raw_eye_points = processed_gaze_data.get('raw_eye_points', [])
                    
                    # 构建平移向量 (兼容 visualizer 可视化接口)
                    tvec = np.array([[off_x], [off_y], [est_dist]])
                    
                    # 姿态重建：根据欧拉角 (Yaw, Pitch, Roll) 重建旋转矩阵
                    # 注：Tracker 现采用几何法直接输出经滤波的 Yaw/Pitch
                    yaw = self.eye_tracker.current_yaw
                    pitch = self.eye_tracker.current_pitch
                    roll = processed_gaze_data.get('roll', 0.0)
                    
                    # 弧度转换
                    y_rad = np.radians(yaw)
                    p_rad = np.radians(pitch)
                    r_rad = np.radians(roll)
                    
                    # 构造各轴旋转矩阵
                    # Rx: 绕 X 轴旋转 (Pitch)
                    Rx = np.array([
                        [1, 0, 0],
                        [0, np.cos(p_rad), -np.sin(p_rad)],
                        [0, np.sin(p_rad), np.cos(p_rad)]
                    ])
                    
                    # Ry: 绕 Y 轴旋转 (Yaw)
                    Ry = np.array([
                        [np.cos(y_rad), 0, np.sin(y_rad)],
                        [0, 1, 0],
                        [-np.sin(y_rad), 0, np.cos(y_rad)]
                    ])
                    
                    # Rz: 绕 Z 轴旋转 (Roll)
                    Rz = np.array([
                        [np.cos(r_rad), -np.sin(r_rad), 0],
                        [np.sin(r_rad), np.cos(r_rad), 0],
                        [0, 0, 1]
                    ])
                    
                    # 组合旋转矩阵 (按 Ry @ Rx 顺序)
                    rmat = Ry @ Rx
                    
                    if VISUALIZE:
                        # 更新可视化数据容器
                        self.gaze_data_container['tvec'] = tvec
                        self.gaze_data_container['rmat'] = rmat
                        # 将旋转矩阵转换为旋转向量 (用于 OpenCV 可视化函数)
                        rvec, _ = cv2.Rodrigues(rmat)
                        self.gaze_data_container['rvec'] = rvec
                        
                        self.latest_gaze_data = self.gaze_data_container
                    
                    try:
                        # 通过 UDP 发送视线追踪数据 (G:距离,X偏移,Y偏移)
                        data_str = f"G:{est_dist:.2f},{off_x:.2f},{off_y:.2f}"
                        self.udp_sender.send(data_str)
                    except Exception as e:
                        print(f"UDP 发送错误: {e}")
            else:
                # 若丢失追踪或正在全帧扫描，重置追踪器状态
                self.eye_tracker.reset()
                
        except queue.Empty:
            pass # 输出队列为空，跳过本帧处理

    def _update_stats(self):
        """每秒更新一次丢包率统计。"""
        self.stats_manager.update_drop_rate()

    def _render(self):
        """
        调用 Visualizer 渲染当前帧的可视化内容。
        包含：FPS、追踪状态、关键点骨架、视线向量等。
        若用户按下退出键 (ESC)，则返回 True。
        """
        if self.current_display_frame is not None and VISUALIZE:
            stats = self.stats_manager.get_stats()
            should_stop = self.visualizer.render(
                self.current_display_frame, 
                roi_info=self.latest_roi_info, 
                eye_points=self.latest_eye_points, 
                raw_eye_points=self.latest_raw_eye_points, 
                tracker=self.eye_tracker, 
                fps=stats['fps'], 
                gaze_data=self.latest_gaze_data,
                hand_result=self.latest_hand_result,
                pose_result=self.latest_pose_result,
                drop_rate=stats['drop_rate'],
                p99_latency=stats.get('p99_latency', 0.0),
                hands_pos=self.latest_hands_pos,
                closest_hand=self.latest_closest_hand,
                using_full_scan=self.latest_using_full_scan
            )
            return should_stop
        else:
            # 非可视化模式下的简单循环控制
            if VISUALIZE:
                if cv2.waitKey(1) & 0xFF == 27:
                    return True
            else:
                time.sleep(0.001) # 降低 CPU 占用
            return False
