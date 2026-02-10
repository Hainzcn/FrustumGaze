
import multiprocessing
import queue
import time
import cv2
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
    def __init__(self, input_queue, output_queue, stop_event, shm_name, frame_shape):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.shm_name = shm_name
        self.frame_shape = frame_shape
        self.daemon = True

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
        
        print("HandProcessorProcess: Started and Ready.")

        while not self.stop_event.is_set():
            try:
                # 阻塞等待任务
                # 任务格式: (frame_id, timestamp_ms)
                # 图像数据直接从共享内存读取
                # 使用 timeout 避免死锁
                task = self.input_queue.get(timeout=0.01)
                frame_id = task['frame_id']
                
                # 从共享内存复制图像数据
                frame = shm_array.copy()
                
                # 降分辨率处理：自适应帧宽高比，将高度降到360像素
                h, w = frame.shape[:2]
                target_h = 360
                scale = target_h / float(h)
                target_w = int(w * scale)
                
                # 转换处理后的帧为 RGB，并进行缩放
                # 注意：MediaPipe 使用归一化坐标，所以不需要手动映射回原始分辨率
                # 除非需要像素坐标，但我们这里直接传递归一化坐标给主进程
                processed_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_rgb = cv2.resize(processed_rgb, (target_w, target_h))
                
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=processed_rgb)
                
                timestamp_ms = int(time.time() * 1000)
                
                # MediaPipe 处理
                detection_result = detector.detect_for_video(mp_image, timestamp_ms)
                
                # 将结果转换为轻量级对象以便传输
                result_lite = HandDetectionResultLite(detection_result.hand_landmarks)
                
                # 将结果放入输出队列
                if self.output_queue.full():
                    try:
                        self.output_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.output_queue.put({
                    'frame_id': frame_id,
                    'hand_result': result_lite,
                    'timestamp': timestamp_ms
                })
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing Error in Hand Process: {e}")
                # import traceback
                # traceback.print_exc()

        # 清理
        detector.close()
        shm_manager.close()
