import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt

LEFT_IRIS = [474,475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Left eye indices list
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
# Right eye indices list
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246 ]

def calculate_mesh_points(width, height, detection_result):
  if len(detection_result.face_landmarks) < 1:
    return None

  mesh_points=np.array([np.multiply([p.x, p.y], [width, height]).astype(int) for p in detection_result.face_landmarks[0]])
  return mesh_points

def draw_iris_on_image(frame, mesh_points, color):
  cv2.polylines(frame, [mesh_points[LEFT_IRIS]], True, color, 1, cv2.LINE_AA)
  cv2.polylines(frame, [mesh_points[RIGHT_IRIS]], True, color, 1, cv2.LINE_AA)

  return frame

def draw_eye_on_image(frame, mesh_points, color):
  cv2.polylines(frame, [mesh_points[LEFT_EYE]], True, color, 1, cv2.LINE_AA)
  cv2.polylines(frame, [mesh_points[RIGHT_EYE]], True, color, 1, cv2.LINE_AA)

  return frame

def get_eye_movement(mesh_points):
  iris_right_corner = mesh_points[RIGHT_IRIS][2]
  eye_right_corner = mesh_points[RIGHT_EYE][0]
  right_diff = iris_right_corner - eye_right_corner

  iris_left_corner = mesh_points[LEFT_IRIS[0]]
  eye_left_corner = mesh_points[LEFT_EYE[8]]
  left_diff = eye_left_corner - iris_left_corner

  axis_x = None
  
  if right_diff[0] > 27:
    axis_x = "left"
  elif left_diff[0] > 27:
    axis_x = "right"
    
  return axis_x
