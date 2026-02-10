
import cv2
import numpy as np
import time
import threading
import queue
import math
import multiprocessing

from config.settings import VISUALIZE, UDP_IP, UDP_PORT, LEFT_IRIS, RIGHT_IRIS, MODEL_POINTS, LEFT_EYE_CENTER_MODEL, RIGHT_EYE_CENTER_MODEL, EYE_RADIUS, AXIS_LENGTH
from modules.camera import CameraModel, ConfigManager, WebcamVideoStream, select_camera_device, select_resolution
from modules.network import UDPSender
from modules.visualizer import Visualizer
from modules.shared_mem import create_shared_array
from utils.image_utils import ImagePreprocessor
from utils.math_utils import get_head_pose, calculate_distance, calculate_single_eye_gaze, calculate_screen_intersection, calculate_weighted_average
from trackers.eye_tracker import EyeTracker
from trackers.face_mesh import FrameProcessorProcess

def main():
    # 启用 multiprocessing 支持 (Windows 下必须)
    multiprocessing.freeze_support()

    # 全局变量用于线程间通信 (这里改为进程间通信)
    # 只能在 main block 内创建 Queue
    input_queue = multiprocessing.Queue(maxsize=2)
    output_queue = multiprocessing.Queue(maxsize=2)
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

    print("Pipeline started: Capture(Thread) -> SharedMem -> Process(Process) -> Main Loop")

    # FPS 计算相关
    prev_frame_time = 0
    last_processed_frame_id = -1
    
    # 本地持有的当前帧副本，用于显示（因为子进程不回传图像）
    current_display_frame = None

    try:
        while True:
            # 1. 从摄像头线程获取最新帧
            has_frame, frame_data = video_stream.read()
            if has_frame:
                frame, frame_id = frame_data
                if frame is not None:
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
                    
                    try:
                        input_queue.put({'frame_id': frame_id}, block=False)
                    except queue.Full:
                        pass

            # 2. 检查是否有处理结果
            try:
                # 非阻塞获取结果
                result_data = output_queue.get_nowait()
                
                # 解析结果
                current_frame_id = result_data['frame_id']
                detection_result = result_data['detection_result']
                roi_info = result_data['roi_info']
                
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
                        # face_landmarks 现在是 simple object list，不是 protobuf
                        # 获取左右虹膜中心
                        pt_left_iris = face_landmarks[468]
                        pt_right_iris = face_landmarks[473]
                        
                        cx_left = int(pt_left_iris.x * w)
                        cy_left = int(pt_left_iris.y * h)
                        
                        cx_right = int(pt_right_iris.x * w)
                        cy_right = int(pt_right_iris.y * h)
                        
                        raw_eye_points = [(cx_left, cy_left), (cx_right, cy_right)]
                        
                        # 滤波
                        eye_points = tracker.filter_eye_points(raw_eye_points)
                        
                        # 计算距离
                        pixel_dist, estimated_dist = calculate_distance(eye_points, w, h, fov=camera_fov)
                        
                        # 计算头部姿态 (注意：face_landmarks 结构变化，get_head_pose 可能需要适配)
                        # get_head_pose 期望的是有 .x, .y 属性的对象，我们的 DetectionResultLite 里的对象满足这个要求
                        pitch, yaw, roll, rvec, tvec = get_head_pose(face_landmarks, w, h, cam_matrix, dist_coeffs, MODEL_POINTS)
                        
                        # 准备视线可视化数据
                        if VISUALIZE and rvec is not None and tvec is not None and current_frame_id % 6 == 0:
                             gaze_data = {
                                 'rvec': rvec,
                                 'tvec': tvec,
                                 'cam_matrix': cam_matrix,
                                 'dist_coeffs': dist_coeffs
                             }

                        # 校正与发送
                        correction_factor = math.cos(math.radians(yaw))
                        if correction_factor < 0.2: correction_factor = 0.2
                        
                        corrected_dist = estimated_dist * correction_factor
                        
                        filtered_pixel, filtered_estimated = tracker.apply_distance_filter(pixel_dist, corrected_dist)
                        tracker.current_pixel_dist = filtered_pixel
                        tracker.current_estimated_dist = filtered_estimated
                        
                        tracker.update_offset(eye_points, w, h, filtered_pixel, filtered_estimated)

                        try:
                            data_str = f"{tracker.current_estimated_dist:.2f},{tracker.current_offset_x:.2f},{tracker.current_offset_y:.2f}"
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
                        gaze_data
                    )
                    if should_stop:
                        break
            
            except queue.Empty:
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
            
        video_stream.stop()
        udp_sender.close()
        shm_manager.close()
        shm_manager.unlink() # 只有创建者 unlink
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
