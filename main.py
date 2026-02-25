
import cv2
import numpy as np
import time
import queue
import multiprocessing
from collections import deque

from config.settings import VISUALIZE, UDP_IP, UDP_PORT, LEFT_IRIS, RIGHT_IRIS, MODEL_POINTS, LEFT_EYE_CENTER_MODEL, RIGHT_EYE_CENTER_MODEL, EYE_RADIUS, AXIS_LENGTH, EYE_TRACKING_INTERVAL, HAND_TRACKING_INTERVAL, EYE_GAZE_CALCULATION_INTERVAL
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
    # 暂时不传入 shm_array，因为需要先确认实际分辨率
    video_stream = WebcamVideoStream(src=camera_index, width=target_w, height=target_h, api_preference=used_api, exposure=exposure_val).start()

    # 等待摄像头预热
    time.sleep(1.0)

    # 读取最终实际分辨率
    actual_w = video_stream.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"摄像头最终实际分辨率: {int(actual_w)}x{int(actual_h)}")

    if int(actual_w) != target_w or int(actual_h) != target_h:
        print(f"警告: 实际分辨率 ({int(actual_w)}x{int(actual_h)}) 与请求分辨率 ({target_w}x{target_h}) 不一致。")

    # 初始化共享内存块 (双缓冲)
    # 创建足够大的共享内存块用于存图像 (Height, Width, 3)
    frame_shape = (int(actual_h), int(actual_w), 3)
    
    shm_names = []
    shm_managers = []
    shm_arrays = []
    
    # 创建两个缓冲区
    for i in range(2):
        name = f"frustum_gaze_frame_buffer_{i}"
        try:
            mgr, arr = create_shared_array(frame_shape, dtype=np.uint8, name=name)
            shm_names.append(name)
            shm_managers.append(mgr)
            shm_arrays.append(arr)
        except Exception as e:
            print(f"Failed to create shared memory {name}: {e}")
            return
        
    # 注入共享内存列表到 VideoStream
    video_stream.set_shared_memory(shm_arrays)

    # 相机模型初始化
    camera_model = CameraModel(actual_w, actual_h, camera_fov)
    cam_matrix = camera_model.cam_matrix
    dist_coeffs = camera_model.dist_coeffs

    # 预创建 gaze_data 字典用于复用 (优化点 1)
    gaze_data_container = {
        'rvec': None,
        'tvec': None,
        'cam_matrix': cam_matrix,
        'dist_coeffs': dist_coeffs,
        'rmat': None
    }

    # 初始化 UDP
    udp_sender = UDPSender(UDP_IP, UDP_PORT)
    
    # 初始化模块
    tracker = EyeTracker()
    preprocessor = ImagePreprocessor() # 这个将传递给子进程
    visualizer = Visualizer()

    # --- 启动处理进程 ---
    # 传递 shm_names 列表
    processing_process = FrameProcessorProcess(
        input_queue, 
        output_queue, 
        preprocessor, 
        stop_event,
        shm_names,
        frame_shape,
        camera_fov=camera_fov
    )
    processing_process.start()

    hand_processing_process = HandProcessorProcess(
        hand_input_queue,
        hand_output_queue,
        stop_event,
        shm_names,
        frame_shape,
        fov=camera_fov
    )
    hand_processing_process.start()

    print("Pipeline started: Capture(Thread) -> SharedMem -> Process(Process) -> Main Loop")

    # FPS 计算相关 (优化：滑动窗口平滑)
    fps_history = deque(maxlen=30) # 存储最近30帧的瞬时 FPS
    prev_frame_time = 0
    last_processed_frame_id = -1
    
    # 本地持有的当前帧副本，用于显示（因为子进程不回传图像）
    current_display_frame = None
    
    # 缓存最新的检测结果
    latest_hand_result = None
    latest_hands_pos = None
    latest_closest_hand = None
    latest_face_result = None
    latest_roi_info = None
    latest_using_full_scan = False # 新增：全图扫描状态
    latest_eye_points = []
    latest_raw_eye_points = []
    latest_gaze_data = None
    latest_fps = 0.0
    
    hand_frame_counter = 0
    eye_frame_counter = 0

    # 丢包计算相关 (优化：统计真正的计算任务丢失)
    # drop_rate 反映的是：本应该被处理的帧，因队列满而被丢弃的比例
    drop_rate = 0.0
    stat_start_time = time.time()
    
    # 统计计数器
    stat_frames_captured = 0      # 摄像头捕获总帧数
    stat_face_tasks_attempted = 0 # 尝试发送给 FaceProcessor 的任务数 (经过 interval 筛选)
    stat_face_tasks_dropped = 0   # 因队列满而丢弃的任务数
    stat_hand_tasks_attempted = 0 # 尝试发送给 HandProcessor 的任务数 (经过 interval 筛选)
    stat_hand_tasks_dropped = 0   # 因队列满而丢弃的任务数
    stat_processed_count = 0      # 实际完成处理并返回结果的帧数
    
    try:
        while True:
            # 1. 从摄像头线程获取最新帧
            # 优化：frame 可能为 None (如果 shm_array 被使用)，因为数据已经直接写入共享内存
            has_frame, frame_data = video_stream.read()
            if has_frame:
                frame, frame_id, buffer_idx = frame_data
                
                # 如果 frame 为 None，说明数据已经在 shm_array 中
                if frame is None and buffer_idx >= 0:
                    # 使用共享内存数据
                    # 为了不污染共享内存（因为子进程要读原始图），显示用的帧必须拷贝
                    current_display_frame = shm_arrays[buffer_idx].copy()
                elif frame is not None:
                    # 传统模式（备用）- 假设只写入第一个buffer
                    np.copyto(shm_arrays[0], frame)
                    current_display_frame = frame
                    buffer_idx = 0 # 强制设为0以便后续逻辑通用
                
                # 记录发送的帧数 (每秒窗口)
                stat_frames_captured += 1
                
                # 通知子进程有新帧
                # 非阻塞 put，如果队列满了就丢弃旧任务（保持实时性）
                
                # 先尝试腾出空间 (可选策略：总是丢弃旧的，或者不丢弃旧的让其满)
                # 这里为了保证实时性，如果满，应该丢弃最旧的，但这需要 get 再 put。
                # 但 get 也会阻塞。这里采用简单的策略：如果满，说明处理慢。
                # 统计逻辑：只有当我们 *尝试* 发送一个新任务，但因满而失败时，才算 Drop。
                
                # 优化计数器防止溢出
                eye_frame_counter = (eye_frame_counter + 1) % EYE_TRACKING_INTERVAL
                if eye_frame_counter == 0:
                    stat_face_tasks_attempted += 1
                    try:
                        input_queue.put({'frame_id': frame_id, 'buffer_idx': buffer_idx}, block=False)
                    except queue.Full:
                        # 队列满，任务被丢弃
                        stat_face_tasks_dropped += 1
                        pass

                # 同样通知手部追踪进程 (每多少帧发送一次，由配置决定)
                hand_frame_counter = (hand_frame_counter + 1) % HAND_TRACKING_INTERVAL
                if hand_frame_counter == 0:
                    stat_hand_tasks_attempted += 1
                    # 统计逻辑：只有当我们 *尝试* 发送一个新任务，但因满而失败时，才算 Drop。
                    try:
                        hand_input_queue.put({'frame_id': frame_id, 'buffer_idx': buffer_idx}, block=False)
                    except queue.Full:
                        # 队列满，任务被丢弃
                        stat_hand_tasks_dropped += 1
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
                    
                    # 无论是否捏起，都发送手掌中心坐标
                    hx = latest_closest_hand.get('x', 0.0)
                    hy = latest_closest_hand.get('y', 0.0)
                    hz = latest_closest_hand.get('z', 0.0)
                    
                    # 格式: H:is_pinching,x,y,z
                    hand_str = f"H:{is_pinching},{hx:.3f},{hy:.3f},{hz:.3f}"
                    udp_sender.send(hand_str)
                    
            except queue.Empty:
                pass

            # 2. 检查是否有处理结果 (人脸)
            try:
                # 非阻塞获取结果
                result_data = output_queue.get_nowait()
                
                # 解析结果
                current_frame_id = result_data['frame_id']
                latest_face_result = result_data['detection_result']
                latest_roi_info = result_data['roi_info']
                latest_using_full_scan = result_data.get('using_full_scan', False) # 获取状态
                
                # 记录处理完成的帧数 (每秒窗口)
                stat_processed_count += 1

                last_processed_frame_id = current_frame_id
                
                # 更新 FPS (使用平滑平均值)
                new_frame_time = time.time()
                if prev_frame_time > 0:
                    delta = new_frame_time - prev_frame_time
                    if delta > 0:
                        instant_fps = 1.0 / delta
                        fps_history.append(instant_fps)
                        
                        # 计算滑动窗口平均值
                        if len(fps_history) > 0:
                            latest_fps = sum(fps_history) / len(fps_history)
                prev_frame_time = new_frame_time
                
                # 处理视线数据 (更新 latest_eye_points 等)
                latest_eye_points = []
                latest_raw_eye_points = []
                latest_gaze_data = None
                
                # 如果是全图扫描模式，则跳过视线计算 (latest_face_result.face_landmarks 为空)
                if not latest_using_full_scan:
                    # 获取预处理后的视线数据
                    processed_gaze_data = result_data.get('processed_gaze_data')
                    
                    if processed_gaze_data:
                        # 更新 tracker 状态以便 Visualizer 使用
                        est_dist, off_x, off_y = processed_gaze_data['gaze_params']
                        tracker.current_estimated_dist = est_dist
                        tracker.current_offset_x = off_x
                        tracker.current_offset_y = off_y
                        tracker.current_pixel_dist = processed_gaze_data.get('current_pixel_dist', 0)
                        tracker.head_center_pos = processed_gaze_data.get('head_center_pos')
                        tracker.current_yaw = processed_gaze_data.get('yaw', 0.0)

                        # 更新最新点位数据
                        latest_eye_points = processed_gaze_data.get('eye_points', [])
                        latest_raw_eye_points = processed_gaze_data.get('raw_eye_points', [])
                        
                        rvec = processed_gaze_data.get('rvec')
                        tvec = processed_gaze_data.get('tvec')
                        
                        # 准备视线可视化数据
                        if VISUALIZE and rvec is not None and tvec is not None:
                            # 优化：复用字典对象，仅更新变化的值
                            gaze_data_container['rvec'] = rvec
                            gaze_data_container['tvec'] = tvec
                            gaze_data_container['rmat'] = processed_gaze_data.get('rmat')
                            latest_gaze_data = gaze_data_container
                        
                        # 发送 UDP (直接使用预计算的值)
                        try:
                            data_str = f"G:{est_dist:.2f},{off_x:.2f},{off_y:.2f}"
                            udp_sender.send(data_str)
                        except Exception as e:
                            print(f"UDP Send Error: {e}")
                    else:
                         # 即使没有 gaze data (可能是 interval 跳过)，也要确保 tracker 状态被重置或保持
                         # 这里我们假设如果没有数据，就不更新 tracker，保持上一帧的状态或者重置
                         # 但如果是一直没有检测到人脸，tracker 应该 reset
                         # 这里的逻辑是：如果是 ROI 模式但没有 gaze data，说明这一帧跳过了计算
                         pass

                else:
                    tracker.reset()
            
            except queue.Empty:
                pass

            # 每秒更新一次丢包率
            current_time = time.time()
            if current_time - stat_start_time >= 1.0:
                # 丢包率 = (因队列满而丢弃的任务数) / (尝试发送的总任务数)
                # 只有当有尝试发送时才计算，否则保持上一秒的值（或者设为0）
                total_attempts = stat_face_tasks_attempted + stat_hand_tasks_attempted
                total_drops = stat_face_tasks_dropped + stat_hand_tasks_dropped
                
                if total_attempts > 0:
                    calculated_drop = total_drops / total_attempts
                    drop_rate = max(0.0, min(1.0, calculated_drop))
                else:
                    drop_rate = 0.0
                
                # 打印调试信息 (可选)
                # print(f"FPS: {latest_fps:.1f} | Captured: {stat_frames_captured} | Attempted(F/H): {stat_face_tasks_attempted}/{stat_hand_tasks_attempted} | Dropped(F/H): {stat_face_tasks_dropped}/{stat_hand_tasks_dropped} | Processed: {stat_processed_count} | DropRate: {drop_rate:.2%}")
                
                # 重置计数器
                stat_frames_captured = 0
                stat_face_tasks_attempted = 0
                stat_face_tasks_dropped = 0
                stat_hand_tasks_attempted = 0
                stat_hand_tasks_dropped = 0
                stat_processed_count = 0
                stat_start_time = current_time

            # 3. 可视化渲染 (解耦：只要有帧就渲染，使用最新的检测结果)
            if current_display_frame is not None and VISUALIZE:
                # 使用 current_display_frame (已是副本) 进行绘制
                frame_to_show = current_display_frame 
                
                should_stop = visualizer.render(
                    frame_to_show, 
                    latest_roi_info, 
                    latest_eye_points, 
                    latest_raw_eye_points, 
                    tracker, 
                    latest_fps, 
                    latest_gaze_data,
                    hand_result=latest_hand_result,
                    drop_rate=drop_rate,
                    hands_pos=latest_hands_pos,
                    closest_hand=latest_closest_hand,
                    using_full_scan=latest_using_full_scan
                )
                if should_stop:
                    break
            else:
                # 即使不渲染，也要处理事件循环以保持响应 (虽然在不显示窗口时意义不大)
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
        for mgr in shm_managers:
            try:
                mgr.close()
                mgr.unlink() # 只有创建者 unlink
            except:
                pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
