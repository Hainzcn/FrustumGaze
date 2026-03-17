import multiprocessing
import queue
import time
import signal
import mediapipe as mp
import numpy as np
from modules.shared_mem import get_shared_array
from utils.image_utils import GlobalImagePreprocessor
from config import settings
from trackers.pose_tracker import PoseTracker, PoseDetectionResultLite
from trackers.common import LandmarkLite

class PoseProcessorProcess(multiprocessing.Process):
    def __init__(self, input_queue, output_queue, stop_event, shm_names, frame_shape, triple_buffer_idx=None):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.shm_names = shm_names # List of names
        self.frame_shape = frame_shape
        self.triple_buffer_idx = triple_buffer_idx # 三缓冲原子索引
        self.daemon = True

    def run(self):
        # 忽略 SIGINT 信号，让主进程处理 Ctrl+C
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        # --- 在子进程中初始化资源 ---
        
        # 1. 连接共享内存 (三缓冲)
        self.shm_managers = []
        self.shm_arrays = []
        
        # 兼容旧代码传入单个 name 的情况
        names = self.shm_names if isinstance(self.shm_names, list) else [self.shm_names]
        
        for name in names:
            try:
                mgr, arr = get_shared_array(name, self.frame_shape)
                self.shm_managers.append(mgr)
                self.shm_arrays.append(arr)
            except Exception as e:
                print(f"PoseProcessorProcess: Failed to connect to shared memory {name}: {e}")
                return

        # 2. 初始化 PoseTracker
        try:
            pose_tracker = PoseTracker()
        except Exception as e:
            print(f"PoseProcessorProcess: 初始化 PoseTracker 失败: {e}")
            return
        
        print(f"PoseProcessorProcess: 进程已启动并就绪。")

        while not self.stop_event.is_set():
            try:
                # 非阻塞检查停止事件，带超时
                if self.stop_event.is_set():
                    break
                
                try:
                    # 阻塞等待任务，设置短超时以便定期检查 stop_event
                    task = self.input_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                frame_id = task['frame_id']
                
                # 三缓冲：始终从最近写完的 buffer 读取
                if self.triple_buffer_idx is not None:
                    read_idx = self.triple_buffer_idx.value
                else:
                    read_idx = task.get('buffer_idx', 0)
                
                if 0 <= read_idx < len(self.shm_arrays):
                    frame = self.shm_arrays[read_idx]
                else:
                    frame = self.shm_arrays[0]
                
                # 获取原始分辨率
                h, w = frame.shape[:2]
                
                # 计算全图模式下的目标分辨率
                (target_w, target_h), _, _ = GlobalImagePreprocessor.calculate_dimensions(frame.shape, settings.PREPROCESS_TARGET_HEIGHT)
                
                # 准备全图图像
                resized_bgr = GlobalImagePreprocessor.resize_image(frame, target_size=(target_w, target_h))
                processed_rgb_full = GlobalImagePreprocessor.to_rgb(resized_bgr)
                processed_rgb_full = GlobalImagePreprocessor.apply_gaussian_blur(processed_rgb_full, kernel_size=settings.PREPROCESS_GAUSSIAN_KERNEL_SIZE, sigma=settings.PREPROCESS_GAUSSIAN_SIGMA)
                
                timestamp_ms = int(time.time() * 1000)
                
                pose_landmarks_out = []
                
                if processed_rgb_full is not None:
                    mp_image_pose = mp.Image(image_format=mp.ImageFormat.SRGB, data=processed_rgb_full)
                    pose_result = pose_tracker.detect(mp_image_pose, timestamp_ms)
                    
                    if pose_result.pose_landmarks:
                        # Extract Shoulders, Elbows
                        # 11: Left Shoulder, 12: Right Shoulder
                        # 13: Left Elbow, 14: Right Elbow
                        landmarks = pose_result.pose_landmarks[0] # Assume single person
                        
                        indices = [11, 12, 13, 14]
                        
                        for idx in indices:
                            lm = landmarks[idx]
                            pose_landmarks_out.append(lm)

                result_lite = PoseDetectionResultLite(pose_landmarks_out)
                
                # 发送结果
                self.output_queue.put({
                    'pose_result': result_lite,
                    'frame_id': frame_id
                })
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"PoseProcessorProcess Error: {e}")
                continue
