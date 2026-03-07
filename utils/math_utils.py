import cv2
import numpy as np
import math
import time

class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def filter(self, x, t=None):
        if t is None:
            t = time.time()
            
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            self.t_prev = t
            return x

        t_e = t - self.t_prev
        
        # Avoid division by zero
        if t_e <= 0.0:
            return self.x_prev

        # The filtered derivative of the signal.
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

class Simple3DKalmanFilter:
    def __init__(self, measurement_noise=0.1, process_noise=0.01):
        # 6 state vars (x, y, z, dx, dy, dz), 3 measurement vars (x, y, z)
        self.kalman = cv2.KalmanFilter(6, 3)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                                  [0, 1, 0, 0, 0, 0],
                                                  [0, 0, 1, 0, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 0, 1, 0, 0],
                                                 [0, 1, 0, 0, 1, 0],
                                                 [0, 0, 1, 0, 0, 1],
                                                 [0, 0, 0, 1, 0, 0],
                                                 [0, 0, 0, 0, 1, 0],
                                                 [0, 0, 0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(6, dtype=np.float32) * process_noise
        self.kalman.measurementNoiseCov = np.eye(3, dtype=np.float32) * measurement_noise
        self.kalman.errorCovPost = np.eye(6, dtype=np.float32)

    def update(self, x, y, z):
        measurement = np.array([[np.float32(x)], [np.float32(y)], [np.float32(z)]])
        self.kalman.correct(measurement)
        prediction = self.kalman.predict()
        return prediction[0][0], prediction[1][0], prediction[2][0]

class OneDKalmanFilter:
    def __init__(self, Q=1e-5, R=0.01):
        self.kf = cv2.KalmanFilter(2, 1)
        self.kf.measurementMatrix = np.array([[1, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)
        # 过程噪声协方差 (Q) - 预测不确定性
        self.kf.processNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * Q
        # 测量噪声协方差 (R) - 测量不确定性
        self.kf.measurementNoiseCov = np.array([[1]], np.float32) * R
        self.kf.statePost = np.array([[0], [0]], np.float32)
        self.first_run = True

    def update(self, measurement):
        if self.first_run:
            self.kf.statePost = np.array([[measurement], [0]], np.float32)
            self.first_run = False
        
        self.kf.predict()
        self.kf.correct(np.array([[measurement]], np.float32))
        return self.kf.statePost[0][0]

class AdaptiveEKF:
    def __init__(self, process_noise=1e-4, measurement_noise_base=1e-3):
        # State: [theta, phi, d_theta, d_phi]
        self.kf = cv2.KalmanFilter(4, 2)
        # Transition Matrix (F)
        # theta_k+1 = theta_k + d_theta_k
        # phi_k+1 = phi_k + d_phi_k
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        
        # Measurement Matrix (H)
        # z = [theta, phi]
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)
        
        # Process Noise Covariance (Q)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        
        # Measurement Noise Covariance (R) - Base value
        self.R_base = np.eye(2, dtype=np.float32) * measurement_noise_base
        self.kf.measurementNoiseCov = self.R_base.copy()
        
        # Initial State
        self.kf.statePost = np.zeros((4, 1), np.float32)
        self.first_run = True

    def predict(self):
        if self.first_run:
            return 0.0, 0.0
        
        prediction = self.kf.predict()
        return prediction[0][0], prediction[1][0]

    def correct(self, theta, phi, q):
        if self.first_run:
            self.kf.statePost = np.array([[theta], [phi], [0], [0]], np.float32)
            self.first_run = False
            return theta, phi

        # Dynamic R adjustment: R = R_base / q^2
        q_clamped = max(q, 0.01)
        scaler = 1.0 / (q_clamped ** 2)
        
        # Check for singularity at poles (theta close to 0 or PI)
        # In these regions, phi is unstable and should not be trusted.
        # We increase R_phi significantly to rely on prediction for phi.
        is_singular = False
        if theta < 0.1 or theta > (math.pi - 0.1):
            is_singular = True
            
        # Create temporary R matrix
        # IMPORTANT: Ensure result is float32 for OpenCV KalmanFilter
        R_temp = (self.R_base * scaler).astype(np.float32)
        
        if is_singular:
            # Increase measurement noise for phi (index 1,1)
            # This makes the filter ignore the unstable phi measurement
            R_temp[1, 1] *= 1000.0
            
        self.kf.measurementNoiseCov = R_temp
        
        # Get predicted phi for wrapping logic (from statePre)
        # Note: predict() MUST be called before correct()
        pred_phi = self.kf.statePre[1][0]
        
        # Handle angle wrapping for phi (azimuth)
        diff = phi - pred_phi
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        phi_adjusted = pred_phi + diff
        
        measurement = np.array([[np.float32(theta)], [np.float32(phi_adjusted)]])
        
        self.kf.correct(measurement)
        
        out_theta = self.kf.statePost[0][0]
        out_phi = self.kf.statePost[1][0]
        
        # Wrap output phi back to [-pi, pi]
        out_phi_wrapped = (out_phi + np.pi) % (2 * np.pi) - np.pi
        
        # Check for NaN output
        if math.isnan(out_theta) or math.isnan(out_phi_wrapped):
            # Reset filter if NaN detected
            self.kf.statePost = np.zeros((4, 1), np.float32)
            self.kf.errorCovPost = np.eye(4, dtype=np.float32)
            self.first_run = True
            return theta, phi

        return out_theta, out_phi_wrapped

    def update(self, theta, phi, q):
        # Convenience method for predict + correct
        self.predict()
        return self.correct(theta, phi, q)

def calculate_screen_intersection(eye_pos, gaze_vec, z_plane=0.0):
    """
    计算视线与屏幕平面 (Z=0) 的交点
    :param eye_pos: 眼球中心/起点 (x, y, z)
    :param gaze_vec: 视线向量 (x, y, z)
    :param z_plane: 屏幕平面的 Z 坐标 (默认为 0)
    :return: intersection_point (x, y, 0) or None
    """
    # 视线必须指向屏幕 (Z 减小的方向)
    if gaze_vec[2] >= 0:
        return None
        
    # P = O + t * D
    # P.z = O.z + t * D.z = z_plane
    # t = (z_plane - O.z) / D.z
    
    t = (z_plane - eye_pos[2]) / gaze_vec[2]
    
    if t < 0:
        return None # 交点在背后
        
    intersection = eye_pos + t * gaze_vec
    return intersection

def calculate_weighted_average(p1, p2, w1=0.5, w2=0.5):
    """
    计算两个点的加权平均
    """
    if p1 is None and p2 is None:
        return None
    if p1 is None:
        return p2
    if p2 is None:
        return p1
        
    total_w = w1 + w2
    if total_w <= 0:
        return (p1 + p2) / 2.0
        
    return (p1 * w1 + p2 * w2) / total_w
