import time
from collections import deque
import numpy as np

class StatsManager:
    """性能统计管理器：负责计算 FPS、任务丢包率及 P99 延迟"""
    def __init__(self, fps_window_size=30, drop_rate_interval=1.0, latency_window_size=100):
        self.fps_window_size = fps_window_size
        self.drop_rate_interval = drop_rate_interval
        self.latency_window_size = latency_window_size
        
        # FPS 计算相关
        self.fps_history = deque(maxlen=fps_window_size)
        self.frame_intervals = deque(maxlen=latency_window_size) # 存储帧间隔(ms)用于计算 P99
        self.prev_frame_time = 0
        self.latest_fps = 0.0
        self.latest_p99_latency = 0.0
        
        # 丢包率计算相关
        self.stat_start_time = time.time()
        self.stat_frames_captured = 0      # 摄像头捕获总帧数
        self.stat_face_tasks_attempted = 0 # 尝试发送给 FaceProcessor 的任务数
        self.stat_face_tasks_dropped = 0   # 因队列满而丢弃的任务数
        self.stat_hand_tasks_attempted = 0 # 尝试发送给 HandProcessor 的任务数
        self.stat_hand_tasks_dropped = 0   # 因队列满而丢弃的任务数
        self.stat_processed_count = 0      # 实际完成处理并返回结果的帧数
        self.drop_rate = 0.0

    def update_fps(self):
        """更新并返回当前 FPS 和 P99 延迟"""
        new_frame_time = time.time()
        if self.prev_frame_time > 0:
            delta = new_frame_time - self.prev_frame_time
            if delta > 0:
                instant_fps = 1.0 / delta
                self.fps_history.append(instant_fps)
                
                # 记录帧间隔 (ms)
                self.frame_intervals.append(delta * 1000.0)
                
                if len(self.fps_history) > 0:
                    self.latest_fps = sum(self.fps_history) / len(self.fps_history)
                
                # 计算 P99 延迟
                if len(self.frame_intervals) > 0:
                    # 使用 numpy 计算 P99，如果数据量大可以考虑采样
                    self.latest_p99_latency = np.percentile(self.frame_intervals, 99)
        
        self.prev_frame_time = new_frame_time
        return self.latest_fps

    def record_captured(self):
        """记录捕获帧数"""
        self.stat_frames_captured += 1

    def record_face_task_attempted(self):
        """记录尝试发送面部追踪任务"""
        self.stat_face_tasks_attempted += 1

    def record_face_task_dropped(self):
        """记录面部追踪任务丢弃"""
        self.stat_face_tasks_dropped += 1

    def record_hand_task_attempted(self):
        """记录尝试发送手部追踪任务"""
        self.stat_hand_tasks_attempted += 1

    def record_hand_task_dropped(self):
        """记录手部追踪任务丢弃"""
        self.stat_hand_tasks_dropped += 1

    def record_processed(self):
        """记录完成处理的帧数"""
        self.stat_processed_count += 1

    def update_drop_rate(self):
        """更新并返回丢包率"""
        current_time = time.time()
        if current_time - self.stat_start_time >= self.drop_rate_interval:
            total_attempts = self.stat_face_tasks_attempted + self.stat_hand_tasks_attempted
            total_drops = self.stat_face_tasks_dropped + self.stat_hand_tasks_dropped
            
            if total_attempts > 0:
                calculated_drop = total_drops / total_attempts
                self.drop_rate = max(0.0, min(1.0, calculated_drop))
            else:
                self.drop_rate = 0.0
            
            # 重置计数器
            self.stat_frames_captured = 0
            self.stat_face_tasks_attempted = 0
            self.stat_face_tasks_dropped = 0
            self.stat_hand_tasks_attempted = 0
            self.stat_hand_tasks_dropped = 0
            self.stat_processed_count = 0
            self.stat_start_time = current_time
        
        return self.drop_rate

    def get_stats(self):
        """获取当前统计数据"""
        return {
            'fps': self.latest_fps,
            'drop_rate': self.drop_rate,
            'p99_latency': self.latest_p99_latency,
            'processed_count': self.stat_processed_count
        }
