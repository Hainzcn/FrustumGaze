
import math
import cv2
import numpy as np

# --- 1€ 滤波器实现 (用于稳定性和响应性) ---
class OneEuroFilter:
    def __init__(self, t0, x0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        """
        min_cutoff: 最小截止频率 (Hz)。越小越平滑，延迟越高。
        beta: 速度系数。越大则高速运动时响应越快（减少延迟），但低速时抖动可能增加。
        d_cutoff: 导数截止频率 (Hz)。通常设为 1.0 Hz。
        """
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = 0.0
        self.t_prev = float(t0)

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def filter(self, t, x):
        t_e = t - self.t_prev
        
        # 避免时间戳重复或乱序导致的问题
        if t_e <= 0.0:
            return self.x_prev

        # 计算导数（速度）并进行低通滤波
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        # 根据速度动态调整截止频率
        # 速度越快，cutoff 越高，过滤越少（减少延迟）
        # 速度越慢，cutoff 越低，过滤越多（减少抖动）
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        # 更新状态
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat

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
