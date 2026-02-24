
import cv2
import numpy as np

class GlobalImagePreprocessor:
    """
    全局图像预处理工具类
    提供静态方法用于缩放、灰度化、色域转换等，供所有模块复用
    """
    
    @staticmethod
    def calculate_dimensions(original_shape, target_height):
        """
        根据目标高度计算新的尺寸，保持宽高比
        :param original_shape: (h, w)
        :param target_height: 目标高度
        :return: (target_w, target_h), scale, aspect_ratio
        """
        h, w = original_shape[:2]
        aspect_ratio = w / float(h)
        scale = target_height / float(h)
        target_w = int(w * scale)
        return (target_w, target_height), scale, aspect_ratio

    @staticmethod
    def crop_by_normalized_roi(image, normalized_roi):
        """
        根据归一化 ROI 裁剪图像
        :param image: 输入图像
        :param normalized_roi: (x_min, y_min, x_max, y_max) 0-1 范围
        :return: (cropped_image, roi_info)
                 roi_info: (x, y, w, h) 像素坐标
                 如果 ROI 无效返回 (None, None)
        """
        if normalized_roi is None:
            return None, None
            
        h, w = image.shape[:2]
        roi_x_min, roi_y_min, roi_x_max, roi_y_max = normalized_roi
        
        roi_x = int(roi_x_min * w)
        roi_y = int(roi_y_min * h)
        roi_w_pixel = int((roi_x_max - roi_x_min) * w)
        roi_h_pixel = int((roi_y_max - roi_y_min) * h)
        
        # 边界检查
        roi_x = max(0, roi_x)
        roi_y = max(0, roi_y)
        roi_w_pixel = min(w - roi_x, roi_w_pixel)
        roi_h_pixel = min(h - roi_y, roi_h_pixel)
        
        if roi_w_pixel > 10 and roi_h_pixel > 10:
            cropped = image[roi_y:roi_y+roi_h_pixel, roi_x:roi_x+roi_w_pixel]
            return cropped, (roi_x, roi_y, roi_w_pixel, roi_h_pixel)
            
        return None, None

    @staticmethod
    def apply_gaussian_blur(image, kernel_size=(5, 5), sigma=0):
        """应用高斯模糊"""
        return cv2.GaussianBlur(image, kernel_size, sigma)

    @staticmethod
    def resize_image(image, target_size=None, scale_factor=None, interpolation=cv2.INTER_LINEAR):
        """
        统一缩放逻辑
        :param image: 输入图像
        :param target_size: (width, height)
        :param scale_factor: 缩放比例 (如果提供了 target_size 则忽略)
        :param interpolation: 插值方法
        :return: 缩放后的图像
        """
        if image is None:
            return None
            
        if target_size is not None:
            return cv2.resize(image, target_size, interpolation=interpolation)
        elif scale_factor is not None:
            if scale_factor == 1.0:
                return image
            return cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=interpolation)
        return image

    @staticmethod
    def to_gray(image):
        """转为灰度图"""
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def to_rgb(image):
        """转为 RGB (MediaPipe 需要)"""
        if len(image.shape) == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    @staticmethod
    def to_lab(image):
        """转为 LAB"""
        return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
    @staticmethod
    def from_lab_to_bgr(image):
        """从 LAB 转回 BGR"""
        return cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

    @staticmethod
    def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        应用 CLAHE (对比度受限自适应直方图均衡化)
        如果是彩色图，只对 L 通道应用
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = GlobalImagePreprocessor.to_lab(image)
            l, a, b = cv2.split(lab)
            l2 = clahe.apply(l)
            lab = cv2.merge((l2, a, b))
            return GlobalImagePreprocessor.from_lab_to_bgr(lab)
        else:
            # Grayscale
            return clahe.apply(image)

class ImagePreprocessor:
    def __init__(self):
        self.last_roi = None # (x, y, w, h)
        self.alpha = 0.7 # ROI 平滑因子 (0-1)
        # self.clahe = None # 移除：改用 GlobalImagePreprocessor

    def process(self, frame, last_landmarks=None, padding_factor=2.0):
        """
        预处理：ROI 裁剪 -> 放大 -> 双边滤波 -> 对比度增强
        返回：processed_frame, roi_info (x, y, w, h, scale)
        """
        h_frame, w_frame = frame.shape[:2]
        
        # 1. 计算 ROI
        roi = self._compute_roi(w_frame, h_frame, last_landmarks, padding_factor)
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
            # 使用全局工具
            crop = GlobalImagePreprocessor.resize_image(crop, target_size=(new_w, new_h))
        
        # 3. 双边滤波 (过于复杂删除)
        
        # 4. 对比度增强 (L 通道 CLAHE)
        # 使用全局工具
        crop = GlobalImagePreprocessor.apply_clahe(crop)
        
        return crop, (x, y, w, h, scale_factor)

    def _compute_roi(self, w_frame, h_frame, landmarks, padding_factor=2.0):
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
            padding = padding_factor
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
