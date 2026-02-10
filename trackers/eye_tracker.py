
import time
import math
from modules.filters import OneEuroFilter, OneDKalmanFilter

class EyeTracker:
    def __init__(self):
        self.current_pixel_dist = 0
        self.current_estimated_dist = 0
        self.current_offset_x = 0
        self.current_offset_y = 0
        
        # 初始化距离平滑卡尔曼滤波器（二级层）
        self.pixel_dist_filter = OneDKalmanFilter(Q=0.1, R=5.0)
        self.real_dist_filter = OneDKalmanFilter(Q=0.1, R=5.0)
        # 偏移量滤波器（二级层）
        self.offset_x_filter = OneDKalmanFilter(Q=0.1, R=0.1)
        self.offset_y_filter = OneDKalmanFilter(Q=0.1, R=0.1)
        
        # 记录主视眼位置用于绘制
        self.dominant_eye_pos = None

        # --- 像素级滤波器 (一级层) ---
        # 使用 OneEuroFilter 在源头消除抖动
        # 参数调整建议：min_cutoff 越小越稳，beta 越大越快
        self.filters_initialized = False
        self.lx_filter = None
        self.ly_filter = None
        self.rx_filter = None
        self.ry_filter = None
    
    def reset(self):
        self.dominant_eye_pos = None
        self.filters_initialized = False
        # 保持当前距离值直到计算出新值以避免闪烁
    
    def filter_eye_points(self, eye_points):
        """
        对原始像素坐标进行滤波
        eye_points: [(lx, ly), (rx, ry)]
        """
        current_time = time.time()
        
        if len(eye_points) < 2:
            return eye_points
            
        lx, ly = eye_points[0]
        rx, ry = eye_points[1]

        if not self.filters_initialized:
            # min_cutoff=0.5: 静态时非常平滑
            # beta=0.1: 移动时有一定响应速度，避免过度延迟
            self.lx_filter = OneEuroFilter(current_time, lx, min_cutoff=0.5, beta=0.1)
            self.ly_filter = OneEuroFilter(current_time, ly, min_cutoff=0.5, beta=0.1)
            self.rx_filter = OneEuroFilter(current_time, rx, min_cutoff=0.5, beta=0.1)
            self.ry_filter = OneEuroFilter(current_time, ry, min_cutoff=0.5, beta=0.1)
            self.filters_initialized = True
            return eye_points
        
        # 应用滤波
        f_lx = self.lx_filter.filter(current_time, lx)
        f_ly = self.ly_filter.filter(current_time, ly)
        f_rx = self.rx_filter.filter(current_time, rx)
        f_ry = self.ry_filter.filter(current_time, ry)
        
        return [(f_lx, f_ly), (f_rx, f_ry)]

    def apply_distance_filter(self, pixel_dist, estimated_dist):
        """应用Kalman滤波处理距离数据"""
        # 数据有效性检查
        if pixel_dist <= 0 or estimated_dist <= 0:
            return self.current_pixel_dist, self.current_estimated_dist
            
        filtered_pixel = self.pixel_dist_filter.update(pixel_dist)
        filtered_estimated = self.real_dist_filter.update(estimated_dist)
        
        return filtered_pixel, filtered_estimated

    def update_offset(self, eye_points, frame_width, frame_height, pixel_dist, real_dist_cm):
        """
        计算并更新右眼（画面左侧）相对于摄像机光轴的物理偏移
        采用针孔成像模型 (Pinhole Camera Model):
        X = Z * (x - cx) / fx
        Y = Z * (y - cy) / fy
        """
        # 如果距离无效，尝试使用缓存
        if real_dist_cm <= 0:
            if self.current_estimated_dist > 0:
                real_dist_cm = self.current_estimated_dist
            else:
                return

        if len(eye_points) == 0:
            return
            
        # 1. 确定右眼坐标 (u, v)
        # 假设 eye_points 中 x 坐标较小的是右眼（画面左侧）
        sorted_eyes = sorted(eye_points, key=lambda p: p[0])
        right_eye = sorted_eyes[0]
        self.dominant_eye_pos = (int(right_eye[0]), int(right_eye[1]))
        u, v = right_eye
        
        # 2. 确定相机内参 (Intrinsics)
        # 主点 (Principal Point) 坐标 (cx, cy)
        # 严格定义为图像中心，消除系统性偏差
        cx = frame_width / 2.0
        cy = frame_height / 2.0
        
        # 焦距 (Focal Length) fx, fy
        # 假设像素是正方形 (square pixels)，即 fx = fy
        # 使用与 calculate_distance 一致的 FOV = 60度 计算焦距
        fov = 60.0
        fov_rad = math.radians(fov)
        # tan(fov/2) = (w/2) / f  =>  f = (w/2) / tan(fov/2)
        f = (frame_width / 2.0) / math.tan(fov_rad / 2.0)
        fx = f
        fy = f
        
        # 3. 获取深度 Z (cm)
        Z = real_dist_cm
        
        # 4. 应用针孔模型公式计算物理偏移 (X, Y)
        # x轴向右为正，y轴向下为正 (符合图像坐标系)
        # 当 u = cx, v = cy 时，结果严格为 0
        real_offset_x = Z * (u - cx) / fx
        real_offset_y = Z * (v - cy) / fy
        
        # 滤波 (Keep Secondary Filter)
        filtered_x = self.offset_x_filter.update(real_offset_x)
        filtered_y = self.offset_y_filter.update(real_offset_y)
        
        # 保留浮点数精度
        self.current_offset_x = filtered_x
        self.current_offset_y = filtered_y
