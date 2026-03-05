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
from trackers.face_mesh import FrameProcessorProcess
from trackers.hand_tracker import HandProcessorProcess
from trackers.pose_tracker import PoseProcessorProcess

class FrustumGazePipeline:
    def __init__(self):
        # 启用 multiprocessing 支持 (Windows 下必须)
        multiprocessing.freeze_support()
        
        # 进程间通信
        self.input_queue = multiprocessing.Queue(maxsize=2)
        self.output_queue = multiprocessing.Queue(maxsize=2)
        self.hand_input_queue = multiprocessing.Queue(maxsize=2)
        self.hand_output_queue = multiprocessing.Queue(maxsize=2)
        self.pose_input_queue = multiprocessing.Queue(maxsize=2)
        self.pose_output_queue = multiprocessing.Queue(maxsize=2)
        self.stop_event = multiprocessing.Event()
        
        # 管理器
        self.config_manager = ConfigManager()
        self.stats_manager = StatsManager()
        self.udp_sender = UDPSender(UDP_IP, UDP_PORT)
        self.visualizer = Visualizer()
        
        # 追踪器
        self.eye_tracker = EyeTracker()
        self.preprocessor = ImagePreprocessor()
        
        # 摄像头和流
        self.camera_index = None
        self.camera_fov = 60.0
        self.video_stream = None
        self.camera_model = None
        
        # 共享内存
        self.shm_names = []
        self.shm_managers = []
        self.shm_arrays = []
        self.frame_shape = None
        
        # 子进程
        self.face_process = None
        self.hand_process = None
        self.pose_process = None
        
        # 状态
        self.running = False
        self.current_display_frame = None
        
        # 检测结果
        self.latest_hand_result = None
        self.latest_hands_pos = None
        self.latest_closest_hand = None
        self.latest_face_result = None
        self.latest_pose_result = None
        self.latest_roi_info = None
        self.latest_using_full_scan = False
        self.latest_eye_points = []
        self.latest_raw_eye_points = []
        self.latest_gaze_data = None
        
        # 帧计数器
        self.hand_frame_counter = 0
        self.pose_frame_counter = 0
        self.eye_frame_counter = 0
        
        # 视线数据容器 (用于复用)
        self.gaze_data_container = {
            'rvec': None,
            'tvec': None,
            'cam_matrix': None,
            'dist_coeffs': None,
            'rmat': None
        }

    def setup(self):
        """初始化摄像头和分辨率"""
        # 1. 摄像头选择逻辑
        self.camera_index, self.camera_fov = select_camera_device(self.config_manager)
        if self.camera_index is None:
            return False

        # 2. 尝试不同的API
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
            print(f"Error: Could not open camera {self.camera_index}")
            return False

        # 3. 设置摄像头参数
        target_w, target_h = select_resolution(cap_temp, self.camera_index, self.config_manager)

        # 4. 获取已保存的曝光配置
        exposure_val = -5.0
        if camera_info and "exposure" in camera_info:
            exposure_val = float(camera_info["exposure"])
            print(f"检测到已保存的曝光配置: {exposure_val}")
        else:
            print(f"使用默认曝光值: {exposure_val}")
            self.config_manager.update_camera(self.camera_index, exposure=exposure_val)

        cap_temp.release()

        # 5. 启动视频流
        print(f"正在启动优化视频流 (MJPEG, 独立线程)...")
        print(f"目标分辨率: {target_w}x{target_h}")
        
        self.video_stream = WebcamVideoStream(
            src=self.camera_index, 
            width=target_w, 
            height=target_h, 
            api_preference=used_api, 
            exposure=exposure_val
        ).start()

        # 等待摄像头预热
        time.sleep(1.0)

        # 6. 读取最终实际分辨率 & 共享内存
        actual_w = self.video_stream.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"摄像头最终实际分辨率: {int(actual_w)}x{int(actual_h)}")

        if int(actual_w) != target_w or int(actual_h) != target_h:
            print(f"警告: 实际分辨率 ({int(actual_w)}x{int(actual_h)}) 与请求分辨率 ({target_w}x{target_h}) 不一致。")

        self.frame_shape = (int(actual_h), int(actual_w), 3)
        
        # 初始化共享内存块 (双缓冲)
        for i in range(2):
            name = f"frustum_gaze_frame_buffer_{i}"
            try:
                mgr, arr = create_shared_array(self.frame_shape, dtype=np.uint8, name=name)
                self.shm_names.append(name)
                self.shm_managers.append(mgr)
                self.shm_arrays.append(arr)
            except Exception as e:
                print(f"Failed to create shared memory {name}: {e}")
                return False
        
        self.video_stream.set_shared_memory(self.shm_arrays)

        # 7. 相机模型初始化
        self.camera_model = CameraModel(actual_w, actual_h, self.camera_fov)
        self.gaze_data_container['cam_matrix'] = self.camera_model.cam_matrix
        self.gaze_data_container['dist_coeffs'] = self.camera_model.dist_coeffs

        return True

    def start_processes(self):
        """启动处理进程"""
        self.face_process = FrameProcessorProcess(
            self.input_queue, 
            self.output_queue, 
            self.preprocessor, 
            self.stop_event,
            self.shm_names,
            self.frame_shape,
            camera_fov=self.camera_fov
        )
        self.face_process.start()

        self.hand_process = HandProcessorProcess(
            self.hand_input_queue,
            self.hand_output_queue,
            self.stop_event,
            self.shm_names,
            self.frame_shape,
            fov=self.camera_fov
        )
        self.hand_process.start()
        
        self.pose_process = PoseProcessorProcess(
            self.pose_input_queue,
            self.pose_output_queue,
            self.stop_event,
            self.shm_names,
            self.frame_shape
        )
        self.pose_process.start()

        print("Pipeline started: Capture(Thread) -> SharedMem -> Process(Process) -> Main Loop")

    def stop(self):
        """释放资源"""
        print("Stopping processes...")
        self.running = False
        self.stop_event.set()
        
        if self.face_process:
            self.face_process.join(timeout=2.0)
            if self.face_process.is_alive():
                self.face_process.terminate()

        if self.hand_process:
            self.hand_process.join(timeout=2.0)
            if self.hand_process.is_alive():
                self.hand_process.terminate()

        if self.pose_process:
            self.pose_process.join(timeout=2.0)
            if self.pose_process.is_alive():
                self.pose_process.terminate()
            
        if self.video_stream:
            self.video_stream.stop()
            
        self.udp_sender.close()
        
        for mgr in self.shm_managers:
            try:
                mgr.close()
                mgr.unlink()
            except:
                pass
        cv2.destroyAllWindows()

    def run(self):
        """主循环"""
        if not self.setup():
            return

        self.start_processes()
        self.running = True
        
        try:
            while self.running:
                self._process_frame()
                self._check_hand_results()
                self._check_pose_results()
                self._check_face_results()
                self._update_stats()
                
                if self._render():
                    break
                
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            self.stop()

    def _process_frame(self):
        """获取最新帧并分发任务"""
        has_frame, frame_data = self.video_stream.read()
        if has_frame:
            frame, frame_id, buffer_idx = frame_data
            
            # 如果 frame 为 None，说明数据已经在 shm_array 中
            if frame is None and buffer_idx >= 0:
                self.current_display_frame = self.shm_arrays[buffer_idx].copy()
            elif frame is not None:
                np.copyto(self.shm_arrays[0], frame)
                self.current_display_frame = frame
                buffer_idx = 0
            
            self.stats_manager.record_captured()
            
            # 分发面部追踪任务
            self.eye_frame_counter = (self.eye_frame_counter + 1) % EYE_TRACKING_INTERVAL
            if self.eye_frame_counter == 0:
                self.stats_manager.record_face_task_attempted()
                try:
                    self.input_queue.put({'frame_id': frame_id, 'buffer_idx': buffer_idx}, block=False)
                except queue.Full:
                    self.stats_manager.record_face_task_dropped()

            # 分发手部追踪任务
            self.hand_frame_counter = (self.hand_frame_counter + 1) % HAND_TRACKING_INTERVAL
            if self.hand_frame_counter == 0:
                self.stats_manager.record_hand_task_attempted()
                try:
                    self.hand_input_queue.put({'frame_id': frame_id, 'buffer_idx': buffer_idx}, block=False)
                except queue.Full:
                    self.stats_manager.record_hand_task_dropped()

            # 分发姿态追踪任务
            self.pose_frame_counter = (self.pose_frame_counter + 1) % POSE_TRACKING_INTERVAL
            if self.pose_frame_counter == 0:
                try:
                    self.pose_input_queue.put({'frame_id': frame_id, 'buffer_idx': buffer_idx}, block=False)
                except queue.Full:
                    pass

    def _check_hand_results(self):
        """检查手部追踪结果"""
        try:
            hand_result_data = self.hand_output_queue.get_nowait()
            self.latest_hand_result = hand_result_data.get('hand_result')
            self.latest_hands_pos = hand_result_data.get('hands_pos')
            self.latest_closest_hand = hand_result_data.get('closest_hand')
            
            # 发送手部数据 (如有最近的手)
            if self.latest_closest_hand:
                is_pinching = 1 if self.latest_closest_hand.get('is_pinching', False) else 0
                hx = self.latest_closest_hand.get('x', 0.0)
                hy = self.latest_closest_hand.get('y', 0.0)
                hz = self.latest_closest_hand.get('z', 0.0)
                
                hand_str = f"H:{is_pinching},{hx:.3f},{hy:.3f},{hz:.3f}"
                self.udp_sender.send(hand_str)
        except queue.Empty:
            pass

    def _check_pose_results(self):
        """检查姿态追踪结果"""
        try:
            pose_result_data = self.pose_output_queue.get_nowait()
            self.latest_pose_result = pose_result_data.get('pose_result')
        except queue.Empty:
            pass

    def _check_face_results(self):
        """检查面部追踪结果"""
        try:
            result_data = self.output_queue.get_nowait()
            
            self.latest_face_result = result_data['detection_result']
            self.latest_roi_info = result_data['roi_info']
            self.latest_using_full_scan = result_data.get('using_full_scan', False)
            
            self.stats_manager.record_processed()
            self.stats_manager.update_fps()
            
            # 处理视线数据
            self.latest_eye_points = []
            self.latest_raw_eye_points = []
            self.latest_gaze_data = None
            
            if not self.latest_using_full_scan:
                processed_gaze_data = result_data.get('processed_gaze_data')
                
                if processed_gaze_data:
                    est_dist, off_x, off_y = processed_gaze_data['gaze_params']
                    self.eye_tracker.current_estimated_dist = est_dist
                    self.eye_tracker.current_offset_x = off_x
                    self.eye_tracker.current_offset_y = off_y
                    self.eye_tracker.current_pixel_dist = processed_gaze_data.get('current_pixel_dist', 0)
                    self.eye_tracker.head_center_pos = processed_gaze_data.get('head_center_pos')
                    self.eye_tracker.current_yaw = processed_gaze_data.get('yaw', 0.0)

                    self.latest_eye_points = processed_gaze_data.get('eye_points', [])
                    self.latest_raw_eye_points = processed_gaze_data.get('raw_eye_points', [])
                    
                    rvec = processed_gaze_data.get('rvec')
                    tvec = processed_gaze_data.get('tvec')
                    
                    if VISUALIZE and rvec is not None and tvec is not None:
                        self.gaze_data_container['rvec'] = rvec
                        self.gaze_data_container['tvec'] = tvec
                        self.gaze_data_container['rmat'] = processed_gaze_data.get('rmat')
                        self.latest_gaze_data = self.gaze_data_container
                    
                    try:
                        data_str = f"G:{est_dist:.2f},{off_x:.2f},{off_y:.2f}"
                        self.udp_sender.send(data_str)
                    except Exception as e:
                        print(f"UDP Send Error: {e}")
            else:
                self.eye_tracker.reset()
                
        except queue.Empty:
            pass

    def _update_stats(self):
        """每秒更新一次丢包率"""
        self.stats_manager.update_drop_rate()

    def _render(self):
        """可视化渲染"""
        if self.current_display_frame is not None and VISUALIZE:
            stats = self.stats_manager.get_stats()
            should_stop = self.visualizer.render(
                self.current_display_frame, 
                self.latest_roi_info, 
                self.latest_eye_points, 
                self.latest_raw_eye_points, 
                self.eye_tracker, 
                stats['fps'], 
                self.latest_gaze_data,
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
            if VISUALIZE:
                if cv2.waitKey(1) & 0xFF == 27:
                    return True
            else:
                time.sleep(0.001)
            return False
