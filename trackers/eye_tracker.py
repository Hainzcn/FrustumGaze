import cv2
import time
import math
import numpy as np
from utils.math_utils import OneEuroFilter, OneDKalmanFilter
import config.settings as settings

"""
眼部与头部追踪计算模块。
主要职责：
- 从 MediaPipe 人脸关键点提取虹膜、眼角等兴趣点。
- 使用几何法估算头部的偏航（Yaw）与俯仰（Pitch）角度。
- 基于双通道策略（宽度与长度）估算头部到相机的深度，并融合得到稳健的距离。
- 基于针孔相机模型计算头部中心相对于相机光轴的物理偏移（X/Y）。
"""

class EyeTracker:
    """
    眼部追踪器类，负责计算头部的 3D 空间位置、姿态以及眼部关键点。
    包含多级滤波、深度估算和物理坐标转换。
    """
    def __init__(self):
        # 核心输出状态
        self.current_pixel_dist = 0
        self.current_estimated_dist = 0
        self.current_offset_x = 0
        self.current_offset_y = 0
        
        # 头部姿态角 (用于可视化与视线修正)
        self.current_yaw = 0.0
        self.current_pitch = 0.0
        
        # 二级滤波：对距离和偏移量应用卡尔曼滤波
        face_config = settings.FILTER_CONFIG['FACE']
        dist_cfg = face_config['DISTANCE']
        off_cfg = face_config['OFFSET']

        self.pixel_dist_filter = OneDKalmanFilter(Q=dist_cfg['process_noise'], R=dist_cfg['measurement_noise'])
        self.real_dist_filter = OneDKalmanFilter(Q=dist_cfg['process_noise'], R=dist_cfg['measurement_noise'])
        self.offset_x_filter = OneDKalmanFilter(Q=off_cfg['process_noise'], R=off_cfg['measurement_noise'])
        self.offset_y_filter = OneDKalmanFilter(Q=off_cfg['process_noise'], R=off_cfg['measurement_noise'])
        
        # 状态追踪
        self.head_center_pos = None # 头部中心在帧中的像素位置
        self.calibrated_width_cm = settings.FACE_REF_WIDTH_CM # 动态校准后的面部参考宽度
        self.calibration_config = settings.FILTER_CONFIG['FACE']['CALIBRATION']
        self.current_depth_details = {} # 深度融合的调试细节
        self.filters = {} # OneEuroFilter 缓存字典
        self.filters_initialized = False

    def reset(self):
        """重置所有滤波器和内部状态"""
        self.head_center_pos = None
        self.filters = {}
        self.filters_initialized = False
        self.current_yaw = 0.0
        self.current_pitch = 0.0

    def _get_filter(self, name, value, current_time, min_cutoff=None, beta=None, d_cutoff=None):
        """获取或创建指定名称的 OneEuroFilter 并应用滤波"""
        if name not in self.filters:
            if min_cutoff is None:
                cfg = settings.FILTER_CONFIG['KEYPOINT']
                min_cutoff, beta, d_cutoff = cfg['min_cutoff'], cfg['beta'], cfg['d_cutoff']
            self.filters[name] = OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        return self.filters[name].filter(value, current_time)

    def _extract_landmark_point(self, landmarks, idx, w, h):
        """从 MediaPipe 关键点列表中提取指定索引点的像素坐标"""
        if idx < len(landmarks):
            p = landmarks[idx]
            return (p.x * w, p.y * h)
        return None

    def _extract_and_filter_iris(self, landmarks, w, h, timestamp):
        """提取并滤波虹膜关键点"""
        iris_l = self._extract_landmark_point(landmarks, 468, w, h)
        iris_r = self._extract_landmark_point(landmarks, 473, w, h)
        if not iris_l or not iris_r: return None, None, None, None
        
        cfg = settings.FILTER_CONFIG['FACE']['IRIS']
        f_iris_l = (self._get_filter('iris_lx', iris_l[0], timestamp, **cfg), 
                    self._get_filter('iris_ly', iris_l[1], timestamp, **cfg))
        f_iris_r = (self._get_filter('iris_rx', iris_r[0], timestamp, **cfg), 
                    self._get_filter('iris_ry', iris_r[1], timestamp, **cfg))
        return iris_l, iris_r, f_iris_l, f_iris_r

    def _calculate_depth(self, landmarks, w, h, timestamp, focal_length, yaw, pitch):
        """
        基于双通道策略（面部宽度与眉鼻长度）估算深度。
        通过面部姿态动态调整各通道权重并进行面部宽度的自动校准。
        """
        # 1. 提取并滤波基准点
        inner_l = self._extract_landmark_point(landmarks, 133, w, h)
        inner_r = self._extract_landmark_point(landmarks, 362, w, h)
        outer_l = self._extract_landmark_point(landmarks, 33, w, h)
        outer_r = self._extract_landmark_point(landmarks, 263, w, h)
        nose_tip = self._extract_landmark_point(landmarks, 1, w, h)
        brow_center = self._extract_landmark_point(landmarks, 168, w, h)

        if any(p is None for p in [inner_l, inner_r, outer_l, outer_r, nose_tip, brow_center]):
            return self.current_estimated_dist

        # 滤波处理
        f_outer_l = (self._get_filter('outer_lx', outer_l[0], timestamp), self._get_filter('outer_ly', outer_l[1], timestamp))
        f_outer_r = (self._get_filter('outer_rx', outer_r[0], timestamp), self._get_filter('outer_ry', outer_r[1], timestamp))
        f_nose = (self._get_filter('nose_tip_x', nose_tip[0], timestamp), self._get_filter('nose_tip_y', nose_tip[1], timestamp))
        f_brow = (self._get_filter('brow_center_x', brow_center[0], timestamp), self._get_filter('brow_center_y', brow_center[1], timestamp))

        # 2. 几何投影深度计算
        d_width_px = math.sqrt((f_outer_l[0] - f_outer_r[0])**2 + (f_outer_l[1] - f_outer_r[1])**2)
        d_length_px = math.sqrt((f_nose[0] - f_brow[0])**2 + (f_nose[1] - f_brow[1])**2)
        
        cos_yaw = max(0.2, abs(math.cos(math.radians(yaw))))
        cos_pitch = max(0.2, abs(math.cos(math.radians(pitch))))

        z_width = (focal_length * self.calibrated_width_cm * cos_yaw) / d_width_px if d_width_px > 0 else 0
        z_length = (focal_length * settings.FACE_REF_LENGTH_CM * cos_pitch) / d_length_px if d_length_px > 0 else 0

        # 3. 宽度动态校准 (仅在姿态稳定时)
        if (abs(yaw) < self.calibration_config['min_valid_yaw'] and 
            abs(pitch) < self.calibration_config['min_valid_pitch'] and z_length > 0):
            alpha = self.calibration_config['width_correction_alpha']
            est_width = (z_length * d_width_px) / (focal_length * cos_yaw)
            self.calibrated_width_cm = (1 - alpha) * self.calibrated_width_cm + alpha * est_width

        # 4. 融合深度
        w_width, w_length = math.pow(cos_yaw, 4), math.pow(cos_pitch, 4)
        total_w = w_width + w_length
        est_dist = (z_width * w_width + z_length * w_length) / total_w if total_w > 0 else self.current_estimated_dist

        self.current_depth_details = {'z_width': z_width, 'z_length': z_length, 
                                     'w_width': w_width/total_w if total_w>0 else 0, 
                                     'w_length': w_length/total_w if total_w>0 else 0, 
                                     'calibrated_width': self.calibrated_width_cm}
        
        # 5. 最终深度滤波
        z_cfg = settings.FILTER_CONFIG['FACE'].get('Z_VAL', settings.FILTER_CONFIG['FACE']['DISTANCE'])
        return self._get_filter('face_z_val', est_dist, timestamp, **z_cfg)

    def process_landmarks(self, face_landmarks, frame_width, frame_height, camera_fov, cam_matrix, dist_coeffs, should_calc_gaze=True):
        """综合处理面部关键点，输出位置、姿态及眼部数据"""
        w, h, ts = frame_width, frame_height, time.time()
        focal_length = cam_matrix[0, 0] if cam_matrix is not None else (w / 2.0) / math.tan(math.radians(camera_fov) / 2.0)
        
        # 1. 虹膜处理
        eye_pts, raw_eye_pts = [], []
        if should_calc_gaze:
            iris_l, iris_r, f_iris_l, f_iris_r = self._extract_and_filter_iris(face_landmarks, w, h, ts)
            if f_iris_l and f_iris_r:
                eye_pts, raw_eye_pts = [f_iris_l, f_iris_r], [iris_l, iris_r]

        # 2. 姿态角计算
        pitch, yaw = self._calculate_face_normal_pose(face_landmarks, w, h)
        y_cfg = settings.FILTER_CONFIG['FACE']['YAW']
        p_cfg = settings.FILTER_CONFIG['FACE'].get('PITCH', y_cfg)
        self.current_yaw = self._get_filter('head_yaw', yaw, ts, **y_cfg)
        self.current_pitch = self._get_filter('head_pitch', pitch, ts, **p_cfg)

        # 3. 深度估算
        self.current_estimated_dist = self._calculate_depth(face_landmarks, w, h, ts, focal_length, self.current_yaw, self.current_pitch)
        self.current_pixel_dist = (focal_length * self.calibrated_width_cm) / self.current_estimated_dist if self.current_estimated_dist > 0 else 0

        # 4. 中心追踪点确定
        center_p = self._extract_landmark_point(face_landmarks, 168, w, h) # 眉心点
        if center_p:
            track_p = (self._get_filter('head_center_x', center_p[0], ts), self._get_filter('head_center_y', center_p[1], ts))
        else:
            # 回退方案：使用右眼内角与外角中点
            il, ir = self._extract_landmark_point(face_landmarks, 133, w, h), self._extract_landmark_point(face_landmarks, 362, w, h)
            track_p = ((il[0]+ir[0])/2.0, (il[1]+ir[1])/2.0) if il and ir else None

        # 5. 更新物理偏移
        if track_p:
            self.update_offset(track_p, w, h, self.current_pixel_dist, self.current_estimated_dist, focal_length=focal_length)
        
        return {
            'eye_points': eye_pts, 'raw_eye_points': raw_eye_pts,
            'yaw': self.current_yaw, 'pitch': self.current_pitch, 'roll': 0.0,
            'calibrated_width': self.calibrated_width_cm
        }

    def _calculate_face_normal_pose(self, face_landmarks, w, h):
        """使用面部特征点几何关系估算姿态角"""
        def get_vec(idx):
            p = face_landmarks[idx]
            return np.array([p.x * w, p.y * h, p.z * w])

        # 使用外眼角、下巴、眉心构造坐标系
        lx, rx, chin, glab = get_vec(33), get_vec(263), get_vec(152), get_vec(168)
        normal = np.cross(rx - lx, chin - glab)
        mag = np.linalg.norm(normal)
        if mag == 0: return 0.0, 0.0
        
        normal /= mag
        pitch = math.degrees(math.atan2(normal[1], normal[2])) + 30.0 # 经验偏置修正
        yaw = math.degrees(math.atan2(normal[0], normal[2]))
        return pitch, yaw

    def calculate_single_eye_gaze(self, iris_center_2d, eye_center_model_3d, tvec, cam_matrix, dist_coeffs, eye_radius=60.0, rmat=None):
        """计算单眼的 3D 视线向量 (近似模型)"""
        p_iris = np.array([iris_center_2d[0], iris_center_2d[1], 1.0])
        ray = np.dot(np.linalg.inv(cam_matrix), p_iris)
        ray /= np.linalg.norm(ray)
        
        z = self.current_estimated_dist
        iris_cam = ray * (z / ray[2]) if ray[2] != 0 else ray * z
        
        T = np.array([self.current_offset_x, self.current_offset_y, self.current_estimated_dist])
        eye_center_cam = np.dot(rmat, eye_center_model_3d) + T
        
        gaze = iris_cam - eye_center_cam
        return gaze / np.linalg.norm(gaze), eye_center_cam

    def update_offset(self, tracking_point, frame_width, frame_height, pixel_dist, real_dist_cm, fov=60.0, focal_length=None):
        """计算头部中心相对于相机光轴的物理偏移量 (单位: cm)"""
        if real_dist_cm <= 0: return
        
        u, v = tracking_point
        self.head_center_pos = (int(u), int(v))
        
        # 针孔相机模型反投影
        cx, cy = frame_width / 2.0, frame_height / 2.0
        fx = fy = focal_length if focal_length else (frame_width / 2.0) / math.tan(math.radians(fov) / 2.0)
        
        # 计算并应用卡尔曼滤波
        self.current_offset_x = self.offset_x_filter.update(real_dist_cm * (u - cx) / fx)
        self.current_offset_y = self.offset_y_filter.update(real_dist_cm * (v - cy) / fy)

    def get_gaze_params(self):
        """获取视线追踪所需的深度与偏移参数"""
        return self.current_estimated_dist, self.current_offset_x, self.current_offset_y

