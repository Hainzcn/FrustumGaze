import cv2
import time
import math
import numpy as np
from dataclasses import dataclass, field
from utils.math_utils import OneEuroFilter, OneDKalmanFilter, calculate_screen_intersection, calculate_weighted_average
import config.settings as settings
from config.settings import EYE_RADIUS

"""
眼部与头部追踪计算模块。
主要职责：
- 从 MediaPipe 人脸关键点提取虹膜、眼角等兴趣点。
- 使用几何法估算头部的偏航（Yaw）与俯仰（Pitch）角度。
- 基于双通道策略（宽度与长度）估算头部到相机的深度，并融合得到稳健的距离。
- 基于针孔相机模型计算头部中心相对于相机光轴的物理偏移（X/Y）。
"""

@dataclass
class GazeResult:
    """
    子进程→主进程的视线追踪结果，作为跨进程传输的唯一数据契约。
    新增字段只需在此处添加，即可自动传播到主进程和可视化层。
    """
    estimated_dist: float = 0.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    pixel_dist: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    head_center_pos: tuple = None
    depth_details: dict = field(default_factory=dict)
    eye_points: list = field(default_factory=list)
    raw_eye_points: list = field(default_factory=list)
    calibrated_width: float = 0.0

    left_gaze_vec: np.ndarray = None
    right_gaze_vec: np.ndarray = None
    left_eye_center_cam: np.ndarray = None
    right_eye_center_cam: np.ndarray = None
    screen_point: tuple = None
    left_confidence: float = 0.0
    right_confidence: float = 0.0
    rmat: np.ndarray = None
    rvec: np.ndarray = None


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
        
        # 缓存常用滤波配置，避免每帧从嵌套 dict 查找
        face_config = settings.FILTER_CONFIG['FACE']
        self._kp_config = settings.FILTER_CONFIG['KEYPOINT']
        self._dist_cfg = face_config['DISTANCE']
        self._off_cfg = face_config['OFFSET']
        self._yaw_cfg = face_config['YAW']
        self._pitch_cfg = face_config.get('PITCH', self._yaw_cfg)
        self._iris_cfg = face_config['IRIS']
        self._z_cfg = face_config.get('Z_VAL', self._dist_cfg)
        self.calibration_config = face_config['CALIBRATION']

        # 二级滤波：对距离和偏移量应用卡尔曼滤波
        self.pixel_dist_filter = OneDKalmanFilter(Q=self._dist_cfg['process_noise'], R=self._dist_cfg['measurement_noise'])
        self.real_dist_filter = OneDKalmanFilter(Q=self._dist_cfg['process_noise'], R=self._dist_cfg['measurement_noise'])
        self.offset_x_filter = OneDKalmanFilter(Q=self._off_cfg['process_noise'], R=self._off_cfg['measurement_noise'])
        self.offset_y_filter = OneDKalmanFilter(Q=self._off_cfg['process_noise'], R=self._off_cfg['measurement_noise'])
        
        # 状态追踪
        self.head_center_pos = None
        self.calibrated_width_cm = settings.FACE_REF_WIDTH_CM
        self.current_depth_details = {}
        self.filters = {}
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
                min_cutoff, beta, d_cutoff = self._kp_config['min_cutoff'], self._kp_config['beta'], self._kp_config['d_cutoff']
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
        
        ic = self._iris_cfg
        f_iris_l = (self._get_filter('iris_lx', iris_l[0], timestamp, **ic), 
                    self._get_filter('iris_ly', iris_l[1], timestamp, **ic))
        f_iris_r = (self._get_filter('iris_rx', iris_r[0], timestamp, **ic), 
                    self._get_filter('iris_ry', iris_r[1], timestamp, **ic))
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
            deviation = abs(est_width - self.calibrated_width_cm) / self.calibrated_width_cm if self.calibrated_width_cm > 0 else 0
            if deviation < self.calibration_config['max_deviation_ratio']:
                self.calibrated_width_cm = (1 - alpha) * self.calibrated_width_cm + alpha * est_width
            ref = settings.FACE_REF_WIDTH_CM
            clamp = self.calibration_config['clamp_ratio']
            self.calibrated_width_cm = max(ref * (1 - clamp), min(ref * (1 + clamp), self.calibrated_width_cm))

        # 4. 融合深度
        power = self.calibration_config.get('weight_power', 2)
        w_width, w_length = math.pow(cos_yaw, power), math.pow(cos_pitch, power)
        total_w = w_width + w_length
        est_dist = (z_width * w_width + z_length * w_length) / total_w if total_w > 0 else self.current_estimated_dist

        self.current_depth_details = {'z_width': z_width, 'z_length': z_length, 
                                     'w_width': w_width/total_w if total_w>0 else 0, 
                                     'w_length': w_length/total_w if total_w>0 else 0, 
                                     'calibrated_width': self.calibrated_width_cm}
        
        # 5. 最终深度滤波
        return self._get_filter('face_z_val', est_dist, timestamp, **self._z_cfg)

    @staticmethod
    def _build_rmat(yaw_deg, pitch_deg):
        """从头部欧拉角构建 Ry @ Rx 旋转矩阵"""
        y_rad = np.radians(yaw_deg)
        p_rad = np.radians(pitch_deg)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(p_rad), -np.sin(p_rad)],
            [0, np.sin(p_rad),  np.cos(p_rad)]
        ])
        Ry = np.array([
            [ np.cos(y_rad), 0, np.sin(y_rad)],
            [0,              1, 0             ],
            [-np.sin(y_rad), 0, np.cos(y_rad)]
        ])
        return Ry @ Rx

    @staticmethod
    def _compute_single_eye_gaze(iris_center_2d, eye_center_2d,
                                 cam_matrix, estimated_dist, eye_radius):
        """
        射线-球面求交法计算单眼视线向量。

        1. 从眼角 landmark 中点推算眼球中心 3D 位置 (球心)
        2. 虹膜射线与眼球球面求交, 得到虹膜表面 3D 点
        3. 视线 = normalize(iris_3d - eye_center)
        """
        cam_inv = np.linalg.inv(cam_matrix)

        # --- 眼球中心 3D: landmark 中点投影到 z=D, 再沿视线后退 r ---
        p_eye = np.array([eye_center_2d[0], eye_center_2d[1], 1.0])
        eye_dir = cam_inv @ p_eye
        eye_dir /= np.linalg.norm(eye_dir)
        eye_surface = eye_dir * (estimated_dist / eye_dir[2])
        C = eye_surface + eye_dir * eye_radius

        # --- 虹膜射线 ---
        p_iris = np.array([iris_center_2d[0], iris_center_2d[1], 1.0])
        d = cam_inv @ p_iris
        d /= np.linalg.norm(d)

        # --- 射线-球面求交: |t*d - C|^2 = r^2 ---
        oc = -C
        half_b = np.dot(d, oc)
        c_val = np.dot(oc, oc) - eye_radius * eye_radius
        discriminant = half_b * half_b - c_val

        if discriminant >= 0:
            sqrt_disc = np.sqrt(discriminant)
            t_front = -half_b - sqrt_disc
            if t_front <= 0:
                t_front = -half_b + sqrt_disc
            iris_3d = d * t_front
        else:
            t_closest = np.dot(d, C)
            iris_3d = d * t_closest

        gaze = iris_3d - C
        norm_val = np.linalg.norm(gaze)
        if norm_val > 0:
            gaze /= norm_val
        return gaze, C

    @staticmethod
    def _compute_eye_confidence(yaw_deg, eye_blink_score, is_left):
        """
        计算单眼追踪置信度，融合几何可见性与眼睑开合度。
        yaw > 0 时左眼更可见，yaw < 0 时右眼更可见。
        """
        k = settings.GAZE_CONFIDENCE_YAW_SENSITIVITY
        min_conf = settings.GAZE_CONFIDENCE_MIN
        yaw_rad = math.radians(yaw_deg)

        sign = 1.0 if is_left else -1.0
        geo = max(min_conf, min(1.0, 0.5 + k * sign * math.sin(yaw_rad)))

        openness = max(0.0, 1.0 - eye_blink_score)
        return geo * openness

    @staticmethod
    def _compute_screen_gaze(l_gaze, l_eye_cam, l_conf,
                             r_gaze, r_eye_cam, r_conf):
        """
        计算双眼视线射线与屏幕平面 (Z=0) 的交点，按置信度加权求最终注视点。
        """
        l_point = calculate_screen_intersection(l_eye_cam, l_gaze) if l_gaze is not None else None
        r_point = calculate_screen_intersection(r_eye_cam, r_gaze) if r_gaze is not None else None

        avg = calculate_weighted_average(l_point, r_point, w1=l_conf, w2=r_conf)
        if avg is not None:
            return (float(avg[0]), float(avg[1]))
        return None

    def process_landmarks(self, face_landmarks, frame_width, frame_height, camera_fov,
                          cam_matrix, dist_coeffs, should_calc_gaze=True,
                          eye_blink_left=0.0, eye_blink_right=0.0):
        """综合处理面部关键点，输出位置、姿态、视线向量及屏幕注视点"""
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
        self.current_yaw = self._get_filter('head_yaw', yaw, ts, **self._yaw_cfg)
        self.current_pitch = self._get_filter('head_pitch', pitch, ts, **self._pitch_cfg)

        # 3. 深度估算
        self.current_estimated_dist = self._calculate_depth(face_landmarks, w, h, ts, focal_length, self.current_yaw, self.current_pitch)
        self.current_pixel_dist = (focal_length * self.calibrated_width_cm) / self.current_estimated_dist if self.current_estimated_dist > 0 else 0

        # 4. 中心追踪点确定
        center_p = self._extract_landmark_point(face_landmarks, 168, w, h)
        if center_p:
            track_p = (self._get_filter('head_center_x', center_p[0], ts), self._get_filter('head_center_y', center_p[1], ts))
        else:
            il, ir = self._extract_landmark_point(face_landmarks, 133, w, h), self._extract_landmark_point(face_landmarks, 362, w, h)
            track_p = ((il[0]+ir[0])/2.0, (il[1]+ir[1])/2.0) if il and ir else None

        # 5. 更新物理偏移
        if track_p:
            self.update_offset(track_p, w, h, self.current_pixel_dist, self.current_estimated_dist, focal_length=focal_length)

        # 6. 视线解算 (仅在 should_calc_gaze 且虹膜可用时执行)
        l_gaze_vec = r_gaze_vec = None
        l_eye_cam = r_eye_cam = None
        screen_pt = None
        l_conf = r_conf = 0.0
        rmat = None
        rvec = None

        if should_calc_gaze and len(eye_pts) == 2 and self.current_estimated_dist > 0:
            rmat = self._build_rmat(self.current_yaw, self.current_pitch)
            rvec, _ = cv2.Rodrigues(rmat)

            # 提取并滤波左右眼角 landmark → 眼球中心 2D
            outer_l = self._extract_landmark_point(face_landmarks, 33, w, h)
            inner_l = self._extract_landmark_point(face_landmarks, 133, w, h)
            outer_r = self._extract_landmark_point(face_landmarks, 263, w, h)
            inner_r = self._extract_landmark_point(face_landmarks, 362, w, h)

            ic = self._iris_cfg
            if outer_l and inner_l:
                f_ol = (self._get_filter('gaze_ol_x', outer_l[0], ts, **ic),
                        self._get_filter('gaze_ol_y', outer_l[1], ts, **ic))
                f_il = (self._get_filter('gaze_il_x', inner_l[0], ts, **ic),
                        self._get_filter('gaze_il_y', inner_l[1], ts, **ic))
                l_eye_center_2d = ((f_ol[0] + f_il[0]) / 2.0, (f_ol[1] + f_il[1]) / 2.0)
            else:
                l_eye_center_2d = eye_pts[0]

            if outer_r and inner_r:
                f_or = (self._get_filter('gaze_or_x', outer_r[0], ts, **ic),
                        self._get_filter('gaze_or_y', outer_r[1], ts, **ic))
                f_ir = (self._get_filter('gaze_ir_x', inner_r[0], ts, **ic),
                        self._get_filter('gaze_ir_y', inner_r[1], ts, **ic))
                r_eye_center_2d = ((f_or[0] + f_ir[0]) / 2.0, (f_or[1] + f_ir[1]) / 2.0)
            else:
                r_eye_center_2d = eye_pts[1]

            l_gaze_vec, l_eye_cam = self._compute_single_eye_gaze(
                eye_pts[0], l_eye_center_2d, cam_matrix,
                self.current_estimated_dist, EYE_RADIUS
            )
            r_gaze_vec, r_eye_cam = self._compute_single_eye_gaze(
                eye_pts[1], r_eye_center_2d, cam_matrix,
                self.current_estimated_dist, EYE_RADIUS
            )

            l_conf = self._compute_eye_confidence(self.current_yaw, eye_blink_left, is_left=True)
            r_conf = self._compute_eye_confidence(self.current_yaw, eye_blink_right, is_left=False)

            screen_pt = self._compute_screen_gaze(
                l_gaze_vec, l_eye_cam, l_conf,
                r_gaze_vec, r_eye_cam, r_conf
            )

        return GazeResult(
            estimated_dist=self.current_estimated_dist,
            offset_x=self.current_offset_x,
            offset_y=self.current_offset_y,
            pixel_dist=self.current_pixel_dist,
            yaw=self.current_yaw,
            pitch=self.current_pitch,
            roll=0.0,
            head_center_pos=self.head_center_pos,
            depth_details=dict(self.current_depth_details) if self.current_depth_details else {},
            eye_points=eye_pts,
            raw_eye_points=raw_eye_pts,
            calibrated_width=self.calibrated_width_cm,
            left_gaze_vec=l_gaze_vec,
            right_gaze_vec=r_gaze_vec,
            left_eye_center_cam=l_eye_cam,
            right_eye_center_cam=r_eye_cam,
            screen_point=screen_pt,
            left_confidence=l_conf,
            right_confidence=r_conf,
            rmat=rmat,
            rvec=rvec,
        )

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

