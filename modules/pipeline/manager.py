import cv2
import numpy as np
import time
import queue
import multiprocessing
import sys
import os
from config.settings import VISUALIZE, UDP_IP, UDP_PORT, EYE_TRACKING_INTERVAL, HAND_TRACKING_INTERVAL, POSE_TRACKING_INTERVAL
from modules.camera import CameraModel, ConfigManager, WebcamVideoStream, select_camera_device, select_resolution
from modules.network import UDPSender
from modules.visualizer import Visualizer
from modules.shared_mem import create_shared_array
from modules.stats import StatsManager
from utils.image_utils import ImagePreprocessor
from trackers.eye_tracker import GazeResult
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
        
        # 追踪器 / 预处理
        self.latest_gaze_result = None # 子进程回传的 GazeResult（取代主进程 EyeTracker 镜像）
        self.preprocessor = ImagePreprocessor() # 图像预处理器
        
        # 摄像头相关属性
        self.camera_index = None # 摄像头索引
        self.camera_device_id = None # 摄像头设备唯一 ID
        self.camera_fov = 60.0 # 摄像头水平视场角 (FOV)
        self.video_stream = None # 视频流对象
        self.camera_model = None # 摄像头模型（包含内参等）
        
        # 共享内存管理 (三缓冲)
        self.shm_names = [] # 共享内存名称列表
        self.shm_managers = [] # 共享内存管理器列表
        self.shm_arrays = [] # 共享 NumPy 数组列表
        self.frame_shape = None # 视频帧的形状 (高, 宽, 通道数)
        self.triple_buffer_idx = multiprocessing.Value('i', 0) # 三缓冲原子索引：指向最近写完的 buffer
        
        # 子进程实例
        self.face_process = None # 人脸处理子进程
        self.hand_process = None # 手部处理子进程
        self.pose_process = None # 姿态处理子进程
        
        # 管道运行状态
        self.running = False # 管道是否正在运行
        self.stopped = False
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

    def _cleanup_shared_memory(self):
        for mgr in self.shm_managers:
            try:
                mgr.close()
                mgr.unlink()
            except Exception as e:
                print(f"清理共享内存失败: {e}")
        self.shm_managers = []
        self.shm_arrays = []
        self.shm_names = []

    def setup(self):
        """
        初始化摄像头并设置分辨率。
        整个流程只 open 一次摄像头——select_camera_device 返回已打开的 cap 句柄，
        经 select_resolution 设置分辨率后直接交给 WebcamVideoStream 接管。
        """
        self.stopped = False
        # 1. 摄像头选择 (包含 API 后端自动测试，返回已打开的 cap 句柄)
        self.camera_index, self.camera_fov, cap, used_api = select_camera_device(self.config_manager)
        if self.camera_index is None:
            print("未选择摄像头，退出。")
            return False
        self.camera_device_id = self.config_manager.get_last_camera()
        if not self.camera_device_id:
            self.camera_device_id = str(self.camera_index)

        # 2. 预设 MJPG 编码后再扫描分辨率，确保扫描结果与最终使用的编码格式一致
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        target_w, target_h = select_resolution(cap, self.camera_index, self.config_manager)

        # 3. 获取并设置曝光配置
        camera_info = self.config_manager.get_camera_info(self.camera_device_id)
        exposure_val = -5.0
        if camera_info and "exposure" in camera_info:
            exposure_val = float(camera_info["exposure"])
            print(f"检测到已保存的曝光配置: {exposure_val}")
        else:
            print(f"使用默认曝光值: {exposure_val}")
            self.config_manager.update_camera(self.camera_device_id, exposure=exposure_val, last_index=self.camera_index)

        # 4. 启动优化视频流 (传递已打开的 cap 句柄，整个流程只 open 一次摄像头)
        print(f"正在启动优化视频流 (MJPEG, 独立线程)...")
        print(f"目标分辨率: {target_w}x{target_h}")
        
        self.video_stream = WebcamVideoStream(
            src=self.camera_index, 
            width=target_w, 
            height=target_h, 
            api_preference=used_api, 
            exposure=exposure_val,
            existing_stream=cap
        ).start()

        # 自适应等待首帧，避免固定 1s 阻塞启动
        warmup_deadline = time.time() + 0.35
        while time.time() < warmup_deadline:
            got_frame, _ = self.video_stream.read()
            if got_frame:
                break
            time.sleep(0.01)

        # 5. 读取实际分辨率并创建共享内存
        actual_w = self.video_stream.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"摄像头最终实际分辨率: {int(actual_w)}x{int(actual_h)}")

        if int(actual_w) != target_w or int(actual_h) != target_h:
            print(f"警告: 实际分辨率 ({int(actual_w)}x{int(actual_h)}) 与请求分辨率 ({target_w}x{target_h}) 不一致。")

        self.frame_shape = (int(actual_h), int(actual_w), 3)
        
        # 初始化共享内存块 (三缓冲机制：写端始终写非 latest buffer，读端从 latest 读取，天然无冲突)
        session_tag = f"{os.getpid()}_{int(time.time() * 1000)}"
        for i in range(3):
            name = f"frustum_gaze_frame_buffer_{session_tag}_{i}"
            try:
                mgr, arr = create_shared_array(self.frame_shape, dtype=np.uint8, name=name)
                self.shm_names.append(name)
                self.shm_managers.append(mgr)
                self.shm_arrays.append(arr)
            except Exception as e:
                print(f"创建共享内存 {name} 失败: {e}")
                self._cleanup_shared_memory()
                return False
        
        self.video_stream.set_shared_memory(self.shm_arrays, self.triple_buffer_idx)

        # 6. 相机模型初始化 (用于姿态和深度计算)
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
            camera_fov=self.camera_fov, # 摄像头视场角
            triple_buffer_idx=self.triple_buffer_idx # 三缓冲原子索引
        )
        self.face_process.start()

        self.hand_process = HandProcessorProcess(
            self.hand_input_queue, # 手部追踪输入队列
            self.hand_output_queue, # 手部追踪输出队列
            self.stop_event, # 停止事件
            self.shm_names, # 共享内存名称
            self.frame_shape, # 帧形状
            fov=self.camera_fov, # 摄像头视场角
            triple_buffer_idx=self.triple_buffer_idx # 三缓冲原子索引
        )
        self.hand_process.start()
        
        self.pose_process = PoseProcessorProcess(
            self.pose_input_queue, # 姿态追踪输入队列
            self.pose_output_queue, # 姿态追踪输出队列
            self.stop_event, # 停止事件
            self.shm_names, # 共享内存名称
            self.frame_shape, # 帧形状
            triple_buffer_idx=self.triple_buffer_idx # 三缓冲原子索引
        )
        self.pose_process.start()

        print("管道启动: 捕获 (线程) -> 共享内存 -> 处理 (进程) -> 主循环")

    def stop(self):
        """
        停止所有子进程、视频流，并释放共享内存和 OpenCV 窗口。
        确保所有资源被正确清理。
        """
        if self.stopped:
            return
        self.stopped = True
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
        self._cleanup_shared_memory()
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
        has_frame, frame_data = self.video_stream.read(timeout=0.033)
        if has_frame:
            frame, frame_id, buffer_idx = frame_data
            
            # 如果 frame 为 None，表示视频流已将帧写入共享内存；否则，回退到直接传帧
            if frame is None and buffer_idx >= 0:
                if VISUALIZE:
                    read_idx = self.triple_buffer_idx.value
                    self.current_display_frame = self.shm_arrays[read_idx].copy()
            elif frame is not None:
                np.copyto(self.shm_arrays[0], frame)
                if VISUALIZE:
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
                is_pinching = 1 if self.latest_closest_hand.get('is_pinching', False) else 0
                pinch_pos = self.latest_closest_hand.get('pinch_pos', (0.0, 0.0, 0.0))

                if is_pinching and any(v != 0.0 for v in pinch_pos):
                    hx, hy, hz = pinch_pos
                else:
                    hx = self.latest_closest_hand.get('x', 0.0)
                    hy = self.latest_closest_hand.get('y', 0.0)
                    hz = self.latest_closest_hand.get('z', 0.0)

                hand_str = f"H:{is_pinching},{hx:.2f},{hy:.2f},{hz:.2f}"
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
        检查面部追踪子进程的输出队列，获取最新的 GazeResult。
        直接使用子进程返回的数据对象，无需逐字段手动同步。
        """
        try:
            result_data = self.output_queue.get_nowait()
            
            self.latest_face_result = result_data['detection_result']
            self.latest_roi_info = result_data['roi_info']
            self.latest_using_full_scan = result_data.get('using_full_scan', False)
            
            self.stats_manager.record_processed()
            self.stats_manager.update_fps()
            
            self.latest_eye_points = []
            self.latest_raw_eye_points = []
            self.latest_gaze_data = None
            
            if not self.latest_using_full_scan:
                gaze = result_data.get('gaze_result')
                
                if gaze:
                    self.latest_gaze_result = gaze
                    self.latest_eye_points = gaze.eye_points
                    self.latest_raw_eye_points = gaze.raw_eye_points
                    
                    if VISUALIZE and gaze.rmat is not None:
                        tvec = np.array([[gaze.offset_x], [gaze.offset_y], [gaze.estimated_dist]])
                        self.gaze_data_container['tvec'] = tvec
                        self.gaze_data_container['rmat'] = gaze.rmat
                        self.gaze_data_container['rvec'] = gaze.rvec
                        self.latest_gaze_data = self.gaze_data_container
                    
                    try:
                        sp = gaze.screen_point
                        sx, sy = (sp[0], sp[1]) if sp else (0.0, 0.0)
                        data_str = f"G:{gaze.estimated_dist:.2f},{gaze.offset_x:.2f},{gaze.offset_y:.2f},{sx:.2f},{sy:.2f}"
                        self.udp_sender.send(data_str)
                    except Exception as e:
                        print(f"UDP 发送错误: {e}")
            else:
                self.latest_gaze_result = None
                
        except queue.Empty:
            pass

    def _update_stats(self):
        """每秒更新一次丢包率与资源占用统计。"""
        self.stats_manager.update_drop_rate()
        self.stats_manager.update_resource_usage()

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
                gaze_result=self.latest_gaze_result, 
                fps=stats['fps'], 
                gaze_data=self.latest_gaze_data,
                hand_result=self.latest_hand_result,
                pose_result=self.latest_pose_result,
                drop_rate=stats['drop_rate'],
                p99_latency=stats.get('p99_latency', 0.0),
                hands_pos=self.latest_hands_pos,
                closest_hand=self.latest_closest_hand,
                using_full_scan=self.latest_using_full_scan,
                resource_stats=stats
            )
            return should_stop
        else:
            if VISUALIZE:
                if cv2.waitKey(1) & 0xFF == 27:
                    return True
            # VISUALIZE=False 时无需 sleep，_process_frame 的阻塞读已让出 CPU
            return False
