
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

class SimpleKalmanFilter:
    def __init__(self, measurement_noise=0.1, process_noise=0.01):
        self.kalman = cv2.KalmanFilter(4, 2) # 4 state vars (x, y, dx, dy), 2 measurement vars (x, y)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)

    def update(self, x, y):
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kalman.correct(measurement)
        prediction = self.kalman.predict()
        return prediction[0][0], prediction[1][0]

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


def calculate_single_eye_gaze(eye_points, eye_model, rvec, tvec, cam_matrix, dist_coeffs, eye_radius=12.0):
    """
    占位符，如果需要从 visualizer.py 迁移相关逻辑，可在此实现。
    """
    pass

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
