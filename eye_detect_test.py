import cv2
import numpy as np
import math

class KalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
        self.kf.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 0.5
    
    def predict(self):
        return self.kf.predict()
    
    def correct(self, measurement):
        if len(measurement.shape) == 3:
            measurement = measurement.reshape(1, 2)
        return self.kf.correct(measurement)

class EyeTracker:
    def __init__(self):
        self.prev_gray = None
        self.track_points = []
        self.track_status = []
        self.kalman_filters = []
        self.detection_interval = 7
        self.frame_count = 0
        self.lost_frames = 0
        self.max_lost_frames = 3
        self.face_rect = None
        self.eye_rects = []
        self.current_pixel_dist = 0
        self.current_estimated_dist = 0
        self.pixel_dist_history = []
        self.estimated_dist_history = []
        self.history_length = 5  # 移动平均窗口大小
    
    def reset(self):
        self.track_points = []
        self.track_status = []
        self.kalman_filters = []
        self.lost_frames = 0
        self.face_rect = None
        self.eye_rects = []
        self.current_pixel_dist = 0
        self.current_estimated_dist = 0
        self.pixel_dist_history = []
        self.estimated_dist_history = []
    
    def update_frame_count(self):
        self.frame_count = (self.frame_count + 1) % self.detection_interval
        return self.frame_count == 0
    
    def is_lost(self):
        return self.lost_frames >= self.max_lost_frames
    
    def increment_lost(self):
        self.lost_frames += 1
    
    def reset_lost(self):
        self.lost_frames = 0
    
    def apply_distance_filter(self, pixel_dist, estimated_dist):
        """应用滤波和除杂处理距离数据"""
        # 数据有效性检查
        if pixel_dist <= 0 or estimated_dist <= 0:
            return self.current_pixel_dist, self.current_estimated_dist
        
        # 限制数据变化幅度（不超过20）
        if self.current_pixel_dist > 0:
            pixel_diff = abs(pixel_dist - self.current_pixel_dist)
            if pixel_diff > 20:
                return self.current_pixel_dist, self.current_estimated_dist
        
        # 添加新数据到历史记录
        self.pixel_dist_history.append(pixel_dist)
        if len(self.pixel_dist_history) > self.history_length:
            self.pixel_dist_history.pop(0)
        
        self.estimated_dist_history.append(estimated_dist)
        if len(self.estimated_dist_history) > self.history_length:
            self.estimated_dist_history.pop(0)
        
        # 计算移动平均值
        filtered_pixel = sum(self.pixel_dist_history) / len(self.pixel_dist_history)
        filtered_estimated = sum(self.estimated_dist_history) / len(self.estimated_dist_history)
        
        # 保留整数
        return int(round(filtered_pixel)), int(round(filtered_estimated))

# 计算眼睛检测质量的函数
def calculate_eye_quality(eye, face_height):
    ex, ey, ew, eh = eye
    pos_score = 1.0 - (ey / (face_height / 2)) if ey < face_height/2 else 0
    size_score = min(ew * eh / (face_height * face_height * 0.1), 1.0)
    aspect_ratio = ew / eh if eh > 0 else 0
    aspect_score = max(0, 1 - abs(aspect_ratio - 2) / 2)
    return (pos_score * 0.4 + size_score * 0.4 + aspect_score * 0.2)

# 选择最佳的两只眼睛
def select_best_eyes(eyes, face_height):
    if len(eyes) <= 2:
        return eyes
    eyes_with_score = [(eye, calculate_eye_quality(eye, face_height)) for eye in eyes]
    eyes_with_score.sort(key=lambda x: x[1], reverse=True)
    best_eyes = [eye for eye, score in eyes_with_score[:2]]
    return best_eyes

# 计算瞳孔间距和距离估算
def calculate_distance(eye_points, frame_width, frame_height, fov=60):
    if len(eye_points) < 2:
        return 0, 0
    
    # 计算瞳孔间距（像素）
    point1, point2 = eye_points[0], eye_points[1]
    pixel_distance = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    # 已知平均瞳距6.5cm
    real_pupil_distance = 6.5  # cm
    
    # 计算摄像头的焦距（基于视角和图像宽度）
    fov_rad = math.radians(fov)
    focal_length = (frame_width / 2) / math.tan(fov_rad / 2)
    
    # 估算距离
    if pixel_distance > 0:
        estimated_distance = (real_pupil_distance * focal_length) / pixel_distance
    else:
        estimated_distance = 0
    
    return pixel_distance, estimated_distance

# 加载Haar级联分类器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
eye_glasses_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

# 初始化跟踪器
tracker = EyeTracker()

# 打开摄像头，尝试不同的API
cap = None
for api in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
    cap = cv2.VideoCapture(0, api)
    if cap.isOpened():
        print(f"Using API: {api}")
        break

if not cap or not cap.isOpened():
    print("Error: Could not open camera")
    exit()

# 设置摄像头参数
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# 等待摄像头初始化
import time
time.sleep(1)

# 测试读取几帧
for i in range(5):
    ret, frame = cap.read()
    if ret:
        break
    time.sleep(0.1)

while True:
    # 读取帧
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        cap.release()
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        time.sleep(0.5)
        if not cap.isOpened():
            print("Error: Could not reopen camera")
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        time.sleep(0.5)
        continue
    
    # 转换为灰度图并预处理
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    # 检查是否需要进行检测或目标丢失
    need_detection = tracker.update_frame_count() or tracker.is_lost()
    
    if need_detection:
        # 重置跟踪器
        tracker.reset()
        
        # 检测人脸，使用更精确的参数
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1,  # 调整缩放因子
            minNeighbors=6,   # 调整邻居数
            minSize=(40, 40),  # 调整最小人脸尺寸
            maxSize=(500, 500), # 调整最大尺寸限制
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # 只处理最大的人脸
        if len(faces) > 0:
            # 按面积排序，选择最大的人脸
            faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
            x, y, w, h = faces[0]
            
            # 绘制人脸矩形
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            tracker.face_rect = (x, y, w, h)
            
            # 提取人脸区域
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # 检测眼睛，使用更精确的参数
            eyes = eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(20, 20),
                maxSize=(w//2, h//3)
            )
            
            # 如果普通分类器没检测到，使用带眼镜的分类器
            if len(eyes) == 0:
                eyes = eye_glasses_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(20, 20),
                    maxSize=(w//2, h//3)
                )
            
            # 过滤眼睛位置
            filtered_eyes = []
            for (ex, ey, ew, eh) in eyes:
                if ey < h/2:
                    filtered_eyes.append((ex, ey, ew, eh))
            
            # 选择最佳的两只眼睛
            best_eyes = select_best_eyes(filtered_eyes, h)
            
            # 初始化跟踪点和卡尔曼滤波器
            eye_points = []
            for (ex, ey, ew, eh) in best_eyes:
                center = (ex + ew//2, ey + eh//2)
                tracker.track_points.append((x + center[0], y + center[1]))
                tracker.kalman_filters.append(KalmanFilter())
                tracker.eye_rects.append((ex, ey, ew, eh))
                eye_points.append((x + center[0], y + center[1]))
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
                cv2.circle(roi_color, (center[0], center[1]), 3, (0, 0, 255), -1)
            
            # 计算瞳孔间距和距离
            if len(eye_points) == 2:
                pixel_dist, estimated_dist = calculate_distance(eye_points, frame.shape[1], frame.shape[0])
                # 应用滤波和除杂
                filtered_pixel, filtered_estimated = tracker.apply_distance_filter(pixel_dist, estimated_dist)
                tracker.current_pixel_dist = filtered_pixel
                tracker.current_estimated_dist = filtered_estimated
            
            tracker.reset_lost()
        else:
            tracker.increment_lost()
    else:
        # 使用光流追踪
        if tracker.prev_gray is not None and len(tracker.track_points) > 0:
            prev_pts = np.array(tracker.track_points, dtype=np.float32).reshape(-1, 1, 2)
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                tracker.prev_gray, gray, prev_pts, None,
                winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            )
            
            tracker.track_status = status.flatten()
            new_track_points = []
            new_kalman_filters = []
            eye_points = []
            
            successful_tracks = 0
            for i, (status, kf) in enumerate(zip(tracker.track_status, tracker.kalman_filters)):
                if status:
                    successful_tracks += 1
                    x, y = next_pts[i][0][0], next_pts[i][0][1]
                    measurement = np.array([[x], [y]], dtype=np.float32)
                    kf.correct(measurement)
                    predicted = kf.predict()
                    new_point = (int(predicted[0][0]), int(predicted[1][0]))
                    new_track_points.append(new_point)
                    new_kalman_filters.append(kf)
                    eye_points.append(new_point)
                    cv2.circle(frame, new_point, 5, (0, 255, 255), -1)
            
            tracker.track_points = new_track_points
            tracker.kalman_filters = new_kalman_filters
            
            # 检查是否丢失目标
            if successful_tracks < 2:
                tracker.increment_lost()
            else:
                # 检查眼睛是否在人脸范围内
                if tracker.face_rect is not None:
                    x, y, w, h = tracker.face_rect
                    in_bounds = True
                    for point in new_track_points:
                        px, py = point
                        # 检查点是否在人脸矩形内
                        if px < x or px > x + w or py < y or py > y + h:
                            in_bounds = False
                            break
                    if not in_bounds:
                        tracker.increment_lost()
                    else:
                        tracker.reset_lost()
                else:
                    tracker.reset_lost()
            
            # 在追踪时也绘制人脸框和眼睛框
            if tracker.face_rect is not None and successful_tracks >= 2:
                x, y, w, h = tracker.face_rect
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # 绘制眼睛框（使用绝对坐标）
                if len(new_track_points) == len(tracker.eye_rects):
                    for i, eye_rect in enumerate(tracker.eye_rects):
                        ex, ey, ew, eh = eye_rect
                        eye_center = new_track_points[i]
                        new_ex = eye_center[0] - ew//2
                        new_ey = eye_center[1] - eh//2
                        cv2.rectangle(frame, (new_ex, new_ey), (new_ex+ew, new_ey+eh), (255, 0, 0), 2)
    
    # 更新前一帧
    tracker.prev_gray = gray.copy()
    
    # 显示距离信息（恒显示）
    if tracker.current_pixel_dist > 0:
        # 检查是否距离过远
        if tracker.current_estimated_dist > 200:
            info_text = "too far!"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            info_text = f"Pupil Distance: {tracker.current_pixel_dist}px | Distance: {tracker.current_estimated_dist}cm"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 绘制瞳孔间距线
            if len(tracker.track_points) >= 2:
                cv2.line(frame, tracker.track_points[0], tracker.track_points[1], (255, 255, 0), 2)
    
    # 添加调试信息
    if tracker.face_rect is not None:
        status_text = f"Tracking: Active | Eyes: {len(tracker.track_points)}"
    else:
        status_text = f"Tracking: Searching | Lost frames: {tracker.lost_frames}/{tracker.max_lost_frames}"
    
    cv2.putText(frame, status_text, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 显示结果
    cv2.imshow('Face and Eye Detection', frame)
    
    # 按ESC键退出
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()