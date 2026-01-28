import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mash = mp_face_mesh.FaceMash(
    max_num_faces = 1,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break



