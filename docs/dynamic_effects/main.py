import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import numpy as np
import cv2 as cv
import time
from sticker import add_glasses, add_hat
from beauty import apply_skin_smooth, apply_whitening, apply_lipstick

model_path = './data/face_landmarker.task'
landmarks_results = None

#options for sticker and beauty effects
ENABLE_GLASSES = True
ENABLE_HAT = False
ENABLE_SMOOTH = False
ENABLE_WHITENING = False
ENABLE_LIPSTICK = False

SMOOTH_STRENGTH = 1 #ranges from 0.0 to 1.0
WHITENING_STRENGTH = 30 #ranges from 0 to much larger values
LIPSTICK_STRENGTH = 0.2 #ranges from 0.0 to 1.0
LIPSTICK_COLOR = (0, 0, 255) #in BGR order

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

def main(ENABLE_GLASSES=ENABLE_GLASSES, ENABLE_HAT=ENABLE_HAT, 
         ENABLE_SMOOTH=ENABLE_SMOOTH, ENABLE_WHITENING=ENABLE_WHITENING, 
         ENABLE_LIPSTICK=ENABLE_WHITENING, frames_num = None):
    global landmarks_results
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
                                 #delegate=python.BaseOptions.Delegate.GPU),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=print_result)

    glasses_png = cv.imread("./data/glasses.png", cv.IMREAD_UNCHANGED)
    hat_png = cv.imread("./data/hat.png", cv.IMREAD_UNCHANGED)
    frame_count = 0
    avg_fps = 0

    with FaceLandmarker.create_from_options(options) as landmarker:
        # The landmarker is initialized. Use it here.

        # Prepare data
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            print("cannot open camera")
            exit()

        start = time.time()
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
            if landmarks_results is not None:
                if ENABLE_GLASSES:
                    display_frame = add_glasses(display_frame, glasses_png, landmarks_results)
                if ENABLE_HAT:
                    display_frame = add_hat(display_frame, hat_png, landmarks_results)
                if ENABLE_SMOOTH:
                    display_frame = apply_skin_smooth(display_frame, landmarks_results, strength=SMOOTH_STRENGTH)
                if ENABLE_WHITENING:
                    display_frame = apply_whitening(display_frame, landmarks_results, strength=WHITENING_STRENGTH)
                if ENABLE_LIPSTICK:
                    display_frame = apply_lipstick(display_frame, landmarks_results, color=LIPSTICK_COLOR, strength=LIPSTICK_STRENGTH)

            frame_count += 1

            cv.imshow('frame', display_frame)

            if cv.waitKey(1) == ord('q'):
                break
            if frames_num is not None and frame_count >= frames_num:
                break

        cap.release()
        cv.destroyAllWindows()
        avg_fps = frame_count / (time.time() - start)
    return avg_fps

if __name__ == "__main__":
    main()