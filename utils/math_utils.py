
import cv2
import numpy as np
import math

def get_head_pose(face_landmarks, w, h, cam_matrix, dist_coeffs, model_points):
    """
    计算头部姿态（Yaw角）用于距离校正
    """
    # 2D 图像点 (使用 MediaPipe 关键点索引)
    # 扩展至 11 个关键点以提高稳定性
    # 1: Nose Tip
    # 152: Chin
    # 33: Left Eye Outer
    # 263: Right Eye Outer
    # 61: Left Mouth Corner
    # 291: Right Mouth Corner
    # 133: Left Eye Inner
    # 362: Right Eye Inner
    # 70: Left Eyebrow Outer
    # 300: Right Eyebrow Outer
    # 2: Nose Bottom
    
    image_points = np.array([
        (face_landmarks[1].x * w, face_landmarks[1].y * h),     # Nose tip
        (face_landmarks[152].x * w, face_landmarks[152].y * h), # Chin
        (face_landmarks[33].x * w, face_landmarks[33].y * h),   # Left eye outer
        (face_landmarks[263].x * w, face_landmarks[263].y * h), # Right eye outer
        (face_landmarks[61].x * w, face_landmarks[61].y * h),   # Left mouth corner
        (face_landmarks[291].x * w, face_landmarks[291].y * h), # Right mouth corner
        (face_landmarks[133].x * w, face_landmarks[133].y * h), # Left eye inner
        (face_landmarks[362].x * w, face_landmarks[362].y * h), # Right eye inner
        (face_landmarks[70].x * w, face_landmarks[70].y * h),   # Left eyebrow outer
        (face_landmarks[300].x * w, face_landmarks[300].y * h), # Right eyebrow outer
        (face_landmarks[2].x * w, face_landmarks[2].y * h)      # Nose bottom
    ], dtype="double")

    # PnP 求解
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)

    if not success:
        return 0, 0, 0, None, None

    # 计算欧拉角
    rmat, jac = cv2.Rodrigues(rotation_vector)
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
    
    # angles[0]=pitch, angles[1]=yaw, angles[2]=roll
    return angles[0], angles[1], angles[2], rotation_vector, translation_vector

def calculate_single_eye_gaze(iris_center_2d, eye_center_model_3d, rvec, tvec, cam_matrix, dist_coeffs, eye_radius=60.0):
    """
    计算单眼视线向量
    :param iris_center_2d: (x, y) 像素坐标
    :param eye_center_model_3d: (x, y, z) 模型坐标系下的眼球中心
    :param rvec: 头部旋转向量
    :param tvec: 头部平移向量
    :param cam_matrix: 相机内参
    :return: (gaze_vector_3d, eye_center_cam_3d) 相机坐标系下的视线向量和眼球中心
    """
    # 1. 将眼球中心变换到相机坐标系
    rmat, _ = cv2.Rodrigues(rvec)
    eye_center_cam = np.dot(rmat, eye_center_model_3d) + tvec.reshape(3)
    
    # 2. 将虹膜 2D 点反投影为射线 (相机坐标系)
    # 先去畸变？为简单起见假设畸变很小或直接使用原始坐标
    # 射线方向: inv(K) * [u, v, 1]
    p_iris_h = np.array([iris_center_2d[0], iris_center_2d[1], 1.0])
    ray_dir = np.dot(np.linalg.inv(cam_matrix), p_iris_h)
    ray_dir = ray_dir / np.linalg.norm(ray_dir) # Normalize
    
    # 3. 计算射线与眼球球面的交点 (或者近似)
    # 球体：圆心 = eye_center_cam，半径 = 60 (12mm * 50 units/cm)
    radius = eye_radius
    
    # 射线: O = (0,0,0), D = ray_dir
    # 交点: |t*D - C|^2 = r^2
    # t^2 - 2(C.D)t + |C|^2 - r^2 = 0
    C = eye_center_cam
    a = 1.0
    b = -2.0 * np.dot(C, ray_dir)
    c_val = np.dot(C, C) - radius**2
    
    discriminant = b**2 - 4*a*c_val
    
    if discriminant >= 0:
        # 射线与球体相交
        t1 = (-b - math.sqrt(discriminant)) / (2*a)
        t2 = (-b + math.sqrt(discriminant)) / (2*a)
        t = min(t1, t2) if t1 > 0 and t2 > 0 else max(t1, t2) # Should be positive
        
        if t > 0:
            iris_on_sphere = t * ray_dir
            gaze_vector = iris_on_sphere - eye_center_cam
            return gaze_vector, eye_center_cam
            
    # Fallback: 如果没有交点（计算误差），使用射线上的最近点
    # 射线上距离 C 最近的点在 t = C.D 处
    t_closest = np.dot(C, ray_dir)
    closest_point = t_closest * ray_dir
    # 强制长度为半径
    vec = closest_point - eye_center_cam
    vec_norm = np.linalg.norm(vec)
    if vec_norm > 0:
        gaze_vector = vec / vec_norm * radius
    else:
        # 回退到头部前方方向 (相机空间中的头部 Z 轴)
        # 相机空间中的头部 Z 轴是 R 的第 3 列
        gaze_vector = rmat[:, 2] * radius
        
    return gaze_vector, eye_center_cam

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

def calculate_distance(eye_points, frame_width, frame_height, fov=55):
    if len(eye_points) < 2:
        return 0, 0
    
    # 计算瞳孔间距（像素）
    point1, point2 = eye_points[0], eye_points[1]
    pixel_distance = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    # 已知平均瞳距6.5cm
    real_pupil_distance = 6  # cm
    
    # 计算摄像头的焦距（基于视角和图像宽度）
    fov_rad = math.radians(fov)
    focal_length = (frame_width / 2) / math.tan(fov_rad / 2)
    
    # 估算距离
    if pixel_distance > 0:
        estimated_distance = (real_pupil_distance * focal_length) / pixel_distance
    else:
        estimated_distance = 0
    
    return pixel_distance, estimated_distance
