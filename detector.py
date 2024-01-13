from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def create_detector(model_path):
    # Load mediapipe's ai model
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                        output_face_blendshapes=True,
                                        output_facial_transformation_matrixes=True,
                                        num_faces=2)

    # Create detector
    detector = vision.FaceLandmarker.create_from_options(options)

    return detector

