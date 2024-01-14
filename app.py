import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from life_cycle import camera_life_cycle
from detector import create_detector

# Options
model_path = 'ai_model.task'
exit_key = 27 

detector = create_detector(model_path)

camera_life_cycle(exit_key, detector)

exit()

# STEP 3: Load the input image.
image = mp.Image.create_from_file("test.jpg")

