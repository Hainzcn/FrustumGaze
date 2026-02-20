
import multiprocessing
import queue
import time
import cv2
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from modules.shared_mem import get_shared_array

# 定义简单的 Landmark 类以便于 Pickle
class LandmarkLite:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

# 定义简单的 Result 类以便于 Pickle
class HandDetectionResultLite:
    def __init__(self, hand_landmarks_list):
        self.multi_hand_landmarks = []
        if hand_landmarks_list:
            for landmarks in hand_landmarks_list:
                simple_landmarks = []
                for lm in landmarks:
                    simple_landmarks.append(LandmarkLite(lm.x, lm.y, lm.z))
                self.multi_hand_landmarks.append(simple_landmarks)

class HandProcessorProcess(multiprocessing.Process):
    def __init__(self, input_queue, output_queue, stop_event, shm_name, frame_shape, fov=60.0):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.shm_name = shm_name
        self.frame_shape = frame_shape
        self.fov = fov
        self.daemon = True

    def _calculate_hand_pos(self, landmarks, aspect_ratio):
        """
        计算手部空间位置 (Camera Space)
        假设: 手掌宽度 (Index MCP 5 -> Pinky MCP 17) 约为 8cm (0.08m)
        """
        HAND_WIDTH_REAL = 0.05  # meters
        
        # 获取关键点
        p5 = landmarks[5]  # INDEX_FINGER_MCP
        p17 = landmarks[17] # PINKY_MCP
        
        # 计算图像平面上的归一化距离 (仅 x, y)
        dx = p5.x - p17.x
        dy = p5.y - p17.y
        w_norm = math.sqrt(dx*dx + dy*dy)
        
        if w_norm < 1e-6:
            return None, None, None, None

        # 计算 Z (深度)
        # Z = W_real / (2 * w_norm * tan(fov/2))
        # 注意: 这里假设 fov 是水平视场角
        tan_half_fov = math.tan(math.radians(self.fov) / 2.0)
        z = HAND_WIDTH_REAL / (2.0 * w_norm * tan_half_fov)
        
        # 计算 X, Y
        # 使用手掌中心 (例如 Index MCP 和 Pinky MCP 的中点，或者 WRIST)
        # 这里使用 5 和 17 的中点作为手掌中心
        cx = (p5.x + p17.x) / 2.0
        cy = (p5.y + p17.y) / 2.0
        
        # X = Z * (cx - 0.5) * 2 * tan(fov/2)
        x = z * (cx - 0.5) * 2.0 * tan_half_fov
        
        # Y = Z * (cy - 0.5) * 2 * tan(fov_v/2)
        # 考虑到 aspect_ratio = W / H
        # tan(fov_v/2) = tan(fov_h/2) / aspect_ratio (近似，或严格推导)
        # 简单推导: Y / Z = (y_pixel - H/2) / f
        # f = (W/2) / tan_half_fov
        # Y = Z * (cy - 0.5) * H / f
        #   = Z * (cy - 0.5) * H * 2 * tan_half_fov / W
        #   = Z * (cy - 0.5) * (1/aspect_ratio) * 2 * tan_half_fov
        y = z * (cy - 0.5) * (1.0 / aspect_ratio) * 2.0 * tan_half_fov
        
        # 坐标系: 
        # X: 右为正
        # Y: 下为正 (OpenCV 默认) -> 也可以转为 上为正 (-y)
        # Z: 前为正
        
        return x, y, z, w_norm

    def run(self):
        # --- 在子进程中初始化资源 ---
        
        # 1. 连接共享内存
        try:
            shm_manager, shm_array = get_shared_array(self.shm_name, self.frame_shape)
        except Exception as e:
            print(f"HandProcessorProcess: Failed to connect to shared memory: {e}")
            return

        # 2. 初始化 MediaPipe Hands (Tasks API)
        try:
            base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                running_mode=vision.RunningMode.VIDEO)
            detector = vision.HandLandmarker.create_from_options(options)
        except Exception as e:
            print(f"HandProcessorProcess: Failed to init MediaPipe: {e}")
            return
        
        print(f"HandProcessorProcess: Started and Ready. FOV={self.fov}")

        while not self.stop_event.is_set():
            try:
                # 阻塞等待任务
                task = self.input_queue.get(timeout=0.01)
                frame_id = task['frame_id']
                
                # 从共享内存复制图像数据
                frame = shm_array.copy()
                
                # 降分辨率处理
                h, w = frame.shape[:2]
                aspect_ratio = w / float(h)
                target_h = 360
                scale = target_h / float(h)
                target_w = int(w * scale)
                
                processed_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_rgb = cv2.resize(processed_rgb, (target_w, target_h))
                
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=processed_rgb)
                timestamp_ms = int(time.time() * 1000)
                
                # MediaPipe 处理
                detection_result = detector.detect_for_video(mp_image, timestamp_ms)
                
                result_lite = HandDetectionResultLite(detection_result.hand_landmarks)
                
                # 计算空间位置并找到最近的手
                closest_hand_info = None
                min_z = float('inf')
                
                # 存储所有手的空间位置，以便 Visualizer 使用
                hands_pos = []

                if result_lite.multi_hand_landmarks:
                    for idx, landmarks in enumerate(result_lite.multi_hand_landmarks):
                        x, y, z, w_norm = self._calculate_hand_pos(landmarks, aspect_ratio)
                        
                        if x is not None:
                            hands_pos.append({'id': idx, 'x': x, 'y': y, 'z': z, 'w_norm': w_norm})
                            
                            if z < min_z:
                                min_z = z
                                closest_hand_info = {'id': idx, 'x': x, 'y': y, 'z': z, 'w_norm': w_norm}

                # 将结果放入输出队列
                if self.output_queue.full():
                    try:
                        self.output_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.output_queue.put({
                    'frame_id': frame_id,
                    'hand_result': result_lite,
                    'timestamp': timestamp_ms,
                    'closest_hand': closest_hand_info,
                    'hands_pos': hands_pos
                })
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing Error in Hand Process: {e}")

        # 清理
        detector.close()
        shm_manager.close()
