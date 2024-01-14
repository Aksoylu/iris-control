import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from helper import calculate_mesh_points, draw_iris_on_image, draw_eye_on_image, get_eye_movement, get_z_index

import pyautogui as mouse
mouse.FAILSAFE = False

SCREEN_SIZE_WIDTH = mouse.size().width
SCREEN_SIZE_HEIGHT = mouse.size().height

MOUSE_X_MOVEMENT_SIZE = (SCREEN_SIZE_WIDTH / 100) * 10
MOUSE_Y_MOVEMENT_SIZE = (SCREEN_SIZE_HEIGHT / 100) * 10

CLICK_THRESHOLD = 5

def camera_life_cycle(exit_key, detector):
    # Get camera device input
    capt = cv2.VideoCapture(0)

    click_indicator = 0
    
    while True:
        # Get image from camera
        _, frame = capt.read()

        width, height = frame.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # Detect face landmarks from the input image.
        detection_result = detector.detect(mp_image)

        z_index = get_z_index(detection_result)

        mesh_points = calculate_mesh_points(height, width, detection_result)
        if mesh_points is None:
            continue

        frame = draw_iris_on_image(frame, mesh_points, (0,0,255))
        frame = draw_eye_on_image(frame, mesh_points, (0,255,0))

        movement = get_eye_movement(mesh_points, z_index)
        mouse_position = mouse.position()

        new_mouse_position_x = mouse_position.x
        new_mouse_position_y = mouse_position.y

        if(movement["axis_x"] == "left"):
            new_mouse_position_x = mouse_position.x - MOUSE_X_MOVEMENT_SIZE
        elif movement["axis_x"] == "right":
            new_mouse_position_x = mouse_position.x + MOUSE_X_MOVEMENT_SIZE
        
        if(movement["axis_y"] == "up"):
            new_mouse_position_y = mouse_position.y - MOUSE_Y_MOVEMENT_SIZE
        if(movement["axis_y"] == "down"):
            new_mouse_position_y = mouse_position.y + MOUSE_Y_MOVEMENT_SIZE

        print(click_indicator)
        if(movement["click"] == True):
            click_indicator += 1
        else:
            click_indicator -= 1
            if click_indicator < 0:
                click_indicator = 0
        
        if click_indicator == CLICK_THRESHOLD:
            click_indicator = 0
            mouse.click()
            print("clicked")

        mouse.moveTo(new_mouse_position_x, new_mouse_position_y)

        
        cv2.imshow("Iris Tracker", frame)
        keyboard_input = cv2.waitKey(30) & 0xff

        if keyboard_input == exit_key:
            break

    capt.release()
    cv2.destroyAllWindows()
