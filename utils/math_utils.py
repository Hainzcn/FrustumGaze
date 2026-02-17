
import cv2
import numpy as np
import math

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
