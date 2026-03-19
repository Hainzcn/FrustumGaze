import cv2
import numpy as np
import math
import time
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# --- Numba Optimized Logic ---

def _one_euro_calc_impl(x, t, x_prev, dx_prev, t_prev, min_cutoff, beta, d_cutoff):
    if t_prev < 0.0:
        return x, 0.0, t
        
    t_e = t - t_prev
    
    if t_e <= 0.0:
        return x_prev, dx_prev, t_prev

    r_d = 2 * math.pi * d_cutoff * t_e
    a_d = r_d / (r_d + 1)
    
    dx = (x - x_prev) / t_e
    dx_hat = a_d * dx + (1 - a_d) * dx_prev

    cutoff = min_cutoff + beta * abs(dx_hat)
    r = 2 * math.pi * cutoff * t_e
    a = r / (r + 1)
    
    x_hat = a * x + (1 - a) * x_prev

    return x_hat, dx_hat, t

if HAS_NUMBA:
    # 启用 cache=True 以减少启动时的编译时间
    # 启用 nogil=True 以在纯数值计算时释放 GIL (虽然在多进程架构下收益有限，但是良好的实践)
    one_euro_calc = jit(nopython=True, cache=True, nogil=True)(_one_euro_calc_impl)
else:
    one_euro_calc = _one_euro_calc_impl

class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = 0.0
        self.dx_prev = 0.0
        self.t_prev = -1.0 # Flag for uninitialized

    def filter(self, x, t=None):
        if t is None:
            t = time.time()
            
        # Ensure inputs are float for numba compatibility
        x_val = float(x)
        t_val = float(t)

        self.x_prev, self.dx_prev, self.t_prev = one_euro_calc(
            x_val, t_val, 
            self.x_prev, self.dx_prev, self.t_prev,
            self.min_cutoff, self.beta, self.d_cutoff
        )
        return self.x_prev

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

    def update(self, x, y, z, R_z=None):
        if R_z is not None:
            self.kalman.measurementNoiseCov[2, 2] = np.float32(R_z)
            
        measurement = np.array([[np.float32(x)], [np.float32(y)], [np.float32(z)]])
        self.kalman.predict()
        self.kalman.correct(measurement)
        state = self.kalman.statePost
        return state[0][0], state[1][0], state[2][0]

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
