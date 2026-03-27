import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import numpy as np
import cv2 as cv
import time
from draw_landmarks import draw_landmarks_on_image

model_path = './data/face_landmarker.task'
landmarks_results = None

# Create the task
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a face landmarker instance with the live stream mode:
def print_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global landmarks_results
    landmarks_results = result

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

with FaceLandmarker.create_from_options(options) as landmarker:
    # The landmarker is initialized. Use it here.

    # Prepare data
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("cannot open camera")
        exit()
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame = cv.flip(frame, 1) #flip the image horizontally
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Convert the frame received from OpenCV to a MediaPipe's Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Get the landmarks of the image
        frame_timestamp_ms = int(time.time() * 1000)
        landmarker.detect_async(mp_image, frame_timestamp_ms)

        display_frame=frame.copy()
        # Draw landmarks on the image
        if landmarks_results is not None:
            annotated_rgb = draw_landmarks_on_image(rgb_frame, landmarks_results)
            display_frame = cv.cvtColor(annotated_rgb, cv.COLOR_RGB2BGR)
        cv.imshow('frame', display_frame)

        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()