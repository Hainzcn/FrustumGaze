
import cv2
import numpy as np

class ImagePreprocessor:
    def __init__(self):
        self.last_roi = None # (x, y, w, h)
        self.alpha = 0.7 # ROI 平滑因子 (0-1)
        self.clahe = None # 延迟初始化以支持 pickle

    def process(self, frame, last_landmarks=None):
        """
        预处理：ROI 裁剪 -> 放大 -> 双边滤波 -> 对比度增强
        返回：processed_frame, roi_info (x, y, w, h, scale)
        """
        if self.clahe is None:
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        h_frame, w_frame = frame.shape[:2]
        
        # 1. 计算 ROI
        roi = self._compute_roi(w_frame, h_frame, last_landmarks)
        x, y, w, h = roi
        
        # 裁剪
        if w <= 0 or h <= 0:
            return frame, (0, 0, w_frame, h_frame, 1.0)
            
        crop = frame[y:y+h, x:x+w]
        
        if crop.size == 0:
            return frame, (0, 0, w_frame, h_frame, 1.0)

        # 2. 像素对齐 / 三次插值放大
        scale_factor = 1.0
        target_min_size = 256
        min_dim = min(w, h)
        
        if min_dim < target_min_size:
            scale_factor = target_min_size / min_dim
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 3. 双边滤波 (边缘保留平滑)
        # (已移除注释代码)
        
        # 4. 对比度增强 (L 通道 CLAHE)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l2 = self.clahe.apply(l)
        lab = cv2.merge((l2, a, b))
        crop = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return crop, (x, y, w, h, scale_factor)

    def _compute_roi(self, w_frame, h_frame, landmarks):
        if landmarks is None:
            # 丢失目标或初始状态：平滑重置为全帧
            target_roi = (0, 0, w_frame, h_frame)
        else:
            # 计算边界框
            xs = [p.x for p in landmarks]
            ys = [p.y for p in landmarks]
            
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            
            # 转换为像素坐标 (中心和尺寸)
            cx = (min_x + max_x) / 2 * w_frame
            cy = (min_y + max_y) / 2 * h_frame
            fw = (max_x - min_x) * w_frame
            fh = (max_y - min_y) * h_frame
            
            # 扩展范围 (填充)
            padding = 2.0 
            size = max(fw, fh) * padding
            
            # 计算左上角
            x = int(cx - size / 2)
            y = int(cy - size / 2)
            w = int(size)
            h = int(size)
            
            # 边界检查
            x = max(0, x)
            y = max(0, y)
            w = min(w, w_frame - x)
            h = min(h, h_frame - y)
            
            target_roi = (x, y, w, h)

        # 平滑 ROI 变化
        if self.last_roi is None:
            self.last_roi = target_roi
            return target_roi
        
        lx, ly, lw, lh = self.last_roi
        tx, ty, tw, th = target_roi
        
        # 如果目标丢失则更快重置
        alpha = self.alpha if landmarks is not None else 0.5
        
        nx = int(lx * alpha + tx * (1 - alpha))
        ny = int(ly * alpha + ty * (1 - alpha))
        nw = int(lw * alpha + tw * (1 - alpha))
        nh = int(lh * alpha + th * (1 - alpha))
        
        # 边界安全检查
        nx = max(0, nx)
        ny = max(0, ny)
        nw = min(nw, w_frame - nx)
        nh = min(nh, h_frame - ny)
        
        self.last_roi = (nx, ny, nw, nh)
        return self.last_roi

    def restore_landmarks(self, detection_result, roi_info, w_frame, h_frame):
        """将归一化局部坐标还原为归一化全局坐标"""
        if not detection_result.face_landmarks:
            return

        roi_x, roi_y, roi_w, roi_h, scale = roi_info
        
        # 如果是全帧且无缩放，则无需处理
        if roi_x == 0 and roi_y == 0 and roi_w == w_frame and roi_h == h_frame and scale == 1.0:
            return

        for face_landmarks in detection_result.face_landmarks:
            for p in face_landmarks:
                # 转换回全局归一化坐标
                # p.x 在裁剪图像中归一化
                # roi_w 是原始图像中裁剪区域的宽度
                # roi_x 是原始图像中的 x 偏移量
                
                new_x = (p.x * roi_w + roi_x) / w_frame
                new_y = (p.y * roi_h + roi_y) / h_frame
                
                p.x = new_x
                p.y = new_y
