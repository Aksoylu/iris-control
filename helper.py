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

Z_ANCHOR = 27

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

def get_z_index(detection_result):
  if(len(detection_result.face_landmarks) < 1):
    return None

  left_iris_index = detection_result.face_landmarks[0][LEFT_IRIS[0]]
  return (left_iris_index.z * 1000)

def get_eye_movement(mesh_points, z_index):
  iris_right_corner = mesh_points[RIGHT_IRIS][2]
  eye_right_corner = mesh_points[RIGHT_EYE][0]
  iris_left_corner = mesh_points[LEFT_IRIS[0]]
  eye_left_corner = mesh_points[LEFT_EYE[8]]

  iris_down_corner = mesh_points[RIGHT_IRIS[3]]
  eye_up_corner =  mesh_points[RIGHT_EYE[12]]

  iris_up_corner = mesh_points[RIGHT_IRIS[1]]
  eye_down_corner =  mesh_points[RIGHT_EYE[4]]

  # click için left (sol)
  eye_bottom_corner = mesh_points[LEFT_EYE[4]]
  eye_top_corner = mesh_points[LEFT_EYE[12]]

  right_diff = (iris_right_corner - eye_right_corner)[0]
  left_diff = (eye_left_corner - iris_left_corner)[0]
  upper_diff = (iris_down_corner - eye_up_corner)[1]
  downer_diff = (eye_down_corner - iris_up_corner)[1]

  eye_closure_diff = (eye_bottom_corner - eye_top_corner)[1]

  axis_x = None
  axis_y = None
  click = False

  if eye_closure_diff < 12:
    click = True
  
  # yakınız
  if z_index < 0:
    if right_diff > Z_ANCHOR +  abs(z_index) * 1.2:
      axis_x = "left"
    elif left_diff > Z_ANCHOR + abs(z_index) * 1.2:
      axis_x = "right"

  if downer_diff < Z_ANCHOR - abs(z_index) * 1.2:
    axis_y = "down"
  
  elif upper_diff > Z_ANCHOR + abs(z_index) * 1.2 and axis_y != "down":
    axis_y = "up"
  

  # uzağız
  elif z_index > 0:
    if right_diff > Z_ANCHOR - abs(z_index):
      axis_x = "left"
    elif left_diff > Z_ANCHOR - abs(z_index):
      axis_x = "right"

    if upper_diff > Z_ANCHOR - abs(z_index):
      axis_y = "up"
    elif downer_diff < Z_ANCHOR and axis_y != "up":
        axis_y = "down"

  return {
    "axis_x": axis_x,
    "axis_y": None,#axis_y,
    "click": click
  }
