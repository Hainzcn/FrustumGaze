
import cv2
import numpy as np
import time
import threading
import queue
import math
import multiprocessing

from config.settings import VISUALIZE, UDP_IP, UDP_PORT, LEFT_IRIS, RIGHT_IRIS, MODEL_POINTS, LEFT_EYE_CENTER_MODEL, RIGHT_EYE_CENTER_MODEL, EYE_RADIUS, AXIS_LENGTH, EYE_TRACKING_INTERVAL, HAND_TRACKING_INTERVAL
from modules.camera import CameraModel, ConfigManager, WebcamVideoStream, select_camera_device, select_resolution
from modules.network import UDPSender
from modules.visualizer import Visualizer
from modules.shared_mem import create_shared_array
from utils.image_utils import ImagePreprocessor
from trackers.eye_tracker import EyeTracker
from trackers.face_mesh import FrameProcessorProcess
from trackers.hand_tracker import HandProcessorProcess

def main():
    # 启用 multiprocessing 支持 (Windows 下必须)
    multiprocessing.freeze_support()

    # 全局变量用于线程间通信 (这里改为进程间通信)
    # 只能在 main block 内创建 Queue
    input_queue = multiprocessing.Queue(maxsize=2)
    output_queue = multiprocessing.Queue(maxsize=2)
    hand_input_queue = multiprocessing.Queue(maxsize=2)
    hand_output_queue = multiprocessing.Queue(maxsize=2)
    stop_event = multiprocessing.Event()
    
    # 初始化管理器
    config_manager = ConfigManager()
    
    # 摄像头选择逻辑
    camera_index, camera_fov = select_camera_device(config_manager)
    if camera_index is None:
        return

    # 打开摄像头，尝试不同的API
    cap_temp = None
    used_api = cv2.CAP_ANY

    for api in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
        cap_temp = cv2.VideoCapture(camera_index, api)
        if cap_temp.isOpened():
            print(f"检测到可用 API: {api}")
            used_api = api
            break

    if not cap_temp or not cap_temp.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return

    # 设置摄像头参数
    target_w, target_h = select_resolution(cap_temp, camera_index, config_manager)

    # 获取已保存的曝光配置
    camera_info = config_manager.get_camera_info(camera_index)
    exposure_val = -5.0
    if camera_info and "exposure" in camera_info:
        exposure_val = float(camera_info["exposure"])
        print(f"检测到已保存的曝光配置: {exposure_val}")
    else:
        print(f"使用默认曝光值: {exposure_val}")
        config_manager.update_camera(camera_index, exposure=exposure_val)

    # 释放临时 cap
    cap_temp.release()

    print(f"正在启动优化视频流 (MJPEG, 独立线程)...")
    print(f"目标分辨率: {target_w}x{target_h}")

    # 初始化多线程视频流 (Producer Thread inside Main Process)
    video_stream = WebcamVideoStream(src=camera_index, width=target_w, height=target_h, api_preference=used_api, exposure=exposure_val).start()

    # 等待摄像头预热
    time.sleep(1.0)

    # 读取最终实际分辨率
    actual_w = video_stream.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"摄像头最终实际分辨率: {int(actual_w)}x{int(actual_h)}")

    if int(actual_w) != target_w or int(actual_h) != target_h:
        print(f"警告: 实际分辨率 ({int(actual_w)}x{int(actual_h)}) 与请求分辨率 ({target_w}x{target_h}) 不一致。")

    # 相机模型初始化
    camera_model = CameraModel(actual_w, actual_h, camera_fov)
    cam_matrix = camera_model.cam_matrix
    dist_coeffs = camera_model.dist_coeffs

    # 初始化 UDP
    udp_sender = UDPSender(UDP_IP, UDP_PORT)
    
    # 初始化模块
    tracker = EyeTracker()
    preprocessor = ImagePreprocessor() # 这个将传递给子进程
    visualizer = Visualizer()

    # --- 共享内存初始化 ---
    # 创建足够大的共享内存块用于存图像 (Height, Width, 3)
    frame_shape = (int(actual_h), int(actual_w), 3)
    shm_name = "frustum_gaze_frame_buffer"
    try:
        shm_manager, shm_array = create_shared_array(frame_shape, dtype=np.uint8, name=shm_name)
    except Exception as e:
        print(f"Failed to create shared memory: {e}")
        return

    # --- 启动处理进程 ---
    processing_process = FrameProcessorProcess(
        input_queue, 
        output_queue, 
        preprocessor, 
        stop_event,
        shm_name,
        frame_shape
    )
    processing_process.start()

    hand_processing_process = HandProcessorProcess(
        hand_input_queue,
        hand_output_queue,
        stop_event,
        shm_name,
        frame_shape,
        fov=camera_fov
    )
    hand_processing_process.start()

    print("Pipeline started: Capture(Thread) -> SharedMem -> Process(Process) -> Main Loop")

    # FPS 计算相关
    prev_frame_time = 0
    last_processed_frame_id = -1
    
    # 本地持有的当前帧副本，用于显示（因为子进程不回传图像）
    current_display_frame = None
    latest_hand_result = None
    latest_hands_pos = None
    latest_closest_hand = None
    hand_frame_counter = 0
    eye_frame_counter = 0

    # 丢包计算相关
    drop_rate = 0.0
    stat_start_time = time.time()
    frames_in_last_sec = 0
    processed_in_last_sec = 0

    try:
        while True:
            # 1. 从摄像头线程获取最新帧
            has_frame, frame_data = video_stream.read()
            if has_frame:
                frame, frame_id = frame_data
                if frame is not None:
                    # 记录发送的帧数 (每秒窗口)
                    frames_in_last_sec += 1
                    
                    # 写入共享内存
                    # 注意：这里简单的直接写入。为了更严谨应该用锁或多缓冲，但对于 30FPS 视频流，
                    # 且只有一个写者，偶尔的读写撕裂通常可接受。
                    # 为了减少撕裂，可以使用 copyto
                    np.copyto(shm_array, frame)
                    
                    # 更新本地显示用的帧副本
                    current_display_frame = frame
                    
                    # 通知子进程有新帧
                    # 非阻塞 put，如果队列满了就丢弃旧任务（保持实时性）
                    if input_queue.full():
                        try:
                            input_queue.get_nowait()
                        except queue.Empty:
                            pass
                    
                    eye_frame_counter += 1
                    if eye_frame_counter % EYE_TRACKING_INTERVAL == 0:
                        try:
                            input_queue.put({'frame_id': frame_id}, block=False)
                        except queue.Full:
                            pass

                    # 同样通知手部追踪进程 (每多少帧发送一次，由配置决定)
                    hand_frame_counter += 1
                    if hand_frame_counter % HAND_TRACKING_INTERVAL == 0:
                        if hand_input_queue.full():
                            try:
                                hand_input_queue.get_nowait()
                            except queue.Empty:
                                pass
                        
                        try:
                            hand_input_queue.put({'frame_id': frame_id}, block=False)
                        except queue.Full:
                            pass

            # 检查是否有手部追踪结果
            try:
                hand_result_data = hand_output_queue.get_nowait()
                latest_hand_result = hand_result_data.get('hand_result')
                latest_hands_pos = hand_result_data.get('hands_pos')
                latest_closest_hand = hand_result_data.get('closest_hand')
                
                # 发送手部数据 (如有最近的手)
                if latest_closest_hand:
                    is_pinching = 1 if latest_closest_hand.get('is_pinching', False) else 0
                    
                    if is_pinching:
                        # 捏起时发送捏起点坐标
                        hx, hy, hz = latest_closest_hand.get('pinch_pos', (0.0, 0.0, 0.0))
                    else:
                        # 未捏起时发送手掌中心坐标
                        hx = latest_closest_hand.get('x', 0.0)
                        hy = latest_closest_hand.get('y', 0.0)
                        hz = latest_closest_hand.get('z', 0.0)
                    
                    # 格式: H:is_pinching,x,y,z
                    hand_str = f"H:{is_pinching},{hx:.3f},{hy:.3f},{hz:.3f}"
                    udp_sender.send(hand_str)
                    
            except queue.Empty:
                pass

            # 2. 检查是否有处理结果
            try:
                # 非阻塞获取结果
                result_data = output_queue.get_nowait()
                
                # 解析结果
                current_frame_id = result_data['frame_id']
                detection_result = result_data['detection_result']
                roi_info = result_data['roi_info']
                
                # 记录处理完成的帧数 (每秒窗口)
                processed_in_last_sec += 1

                # 每秒更新一次丢包率
                current_time = time.time()
                if current_time - stat_start_time >= 1.0:
                    if frames_in_last_sec > 0:
                        # 丢包率 = (输入帧数 - 处理帧数) / 输入帧数
                        # 限制范围 [0, 1]
                        calculated_drop = (frames_in_last_sec - processed_in_last_sec) / frames_in_last_sec
                        drop_rate = max(0.0, min(1.0, calculated_drop))
                    else:
                        drop_rate = 0.0
                    
                    # 重置计数器
                    frames_in_last_sec = 0
                    processed_in_last_sec = 0
                    stat_start_time = current_time

                last_processed_frame_id = current_frame_id

                # 如果没有显示帧，跳过
                if current_display_frame is None:
                    continue
                    
                frame = current_display_frame # 使用最新的帧进行绘制（可能会有轻微延迟不对齐，但响应快）
                h, w = frame.shape[:2]
                
                # 计算 FPS
                new_frame_time = time.time()
                fps = 0
                if prev_frame_time > 0:
                    delta = new_frame_time - prev_frame_time
                    if delta > 0:
                        fps = 1.0 / delta
                prev_frame_time = new_frame_time
                
                eye_points = []
                raw_eye_points = []
                gaze_data = None
                
                if detection_result.face_landmarks:
                    for face_landmarks in detection_result.face_landmarks:
                        # 使用 EyeTracker 处理所有逻辑
                        results = tracker.process_landmarks(
                            face_landmarks, w, h, camera_fov, cam_matrix, dist_coeffs
                        )
                        
                        if results is None:
                            continue

                        eye_points = results['eye_points']
                        raw_eye_points = results['raw_eye_points']
                        rvec = results['rvec']
                        tvec = results['tvec']
                        
                        # 准备视线可视化数据
                        if VISUALIZE and rvec is not None and tvec is not None and current_frame_id % 6 == 0:
                            gaze_data = {
                                'rvec': rvec,
                                'tvec': tvec,
                                'cam_matrix': cam_matrix,
                                'dist_coeffs': dist_coeffs
                            }

                        try:
                            data_str = f"G:{tracker.current_estimated_dist:.2f},{tracker.current_offset_x:.2f},{tracker.current_offset_y:.2f}"
                            udp_sender.send(data_str)
                        except Exception as e:
                            print(f"UDP Send Error: {e}")
                else:
                    tracker.reset()

                # 3. 可视化渲染
                if VISUALIZE:
                    should_stop = visualizer.render(
                        frame, 
                        roi_info, 
                        eye_points, 
                        raw_eye_points, 
                        tracker, 
                        fps, 
                        gaze_data,
                        hand_result=latest_hand_result,
                        drop_rate=drop_rate,
                        hands_pos=latest_hands_pos,
                        closest_hand=latest_closest_hand
                    )
                    if should_stop:
                        break
            
            except queue.Empty:
                # 没有新结果，但可能需要刷新画面（如果只有手部数据更新了或者只是想保持画面流畅）
                if VISUALIZE and current_display_frame is not None:
                    # 如果只有手部更新了，或者只是为了显示最新的帧
                    # 这里简单起见，只有在有 face 结果时才重绘。
                    # 为了更流畅，即使没有 face result，如果有 frame 更新也应该重绘。
                    # 但现在的架构是依赖 face result 的 roi_info 来重绘。
                    # 如果想完全解耦，需要保存上一次的 roi_info
                    pass

                # 没有新结果，稍微 sleep 避免死循环占用 CPU，或者处理 GUI 事件
                if VISUALIZE:
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                else:
                    time.sleep(0.001)

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # 释放资源
        print("Stopping processes...")
        stop_event.set()
        
        # 给子进程一点时间退出
        processing_process.join(timeout=2.0)
        if processing_process.is_alive():
            processing_process.terminate()

        hand_processing_process.join(timeout=2.0)
        if hand_processing_process.is_alive():
            hand_processing_process.terminate()
            
        video_stream.stop()
        udp_sender.close()
        shm_manager.close()
        shm_manager.unlink() # 只有创建者 unlink
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
