import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import numpy as np
import cv2 as cv


def overlay_sticker(background, sticker, x, y):
    bg_h, bg_w = background.shape[:2]
    st_h, st_w = sticker.shape[:2]

    if x >= bg_w or y >= bg_h or x+st_w <= 0 or y+st_h <= 0:
        return background
    
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(x+st_w, bg_w)
    y2 = min(y+st_h, bg_h)

    #crop sticker
    st_x1 = x1 - x
    st_y1 = y1 - y
    st_x2 = st_x1 + (x2-x1)
    st_y2 = st_y1 + (y2-y1)

    sticker_crop = sticker[st_y1:st_y2, st_x1:st_x2]
    bg_crop = background[y1:y2, x1:x2].astype(np.float32)
    
    sticker_rgb = sticker_crop[:, :, :3].astype(np.float32)
    alpha = sticker_crop[:, :, 3].astype(np.float32) / 255.0
    alpha = alpha[:, :, None]

    blended = alpha * sticker_rgb + (1-alpha) * bg_crop
    background[y1:y2, x1:x2] = blended.astype(np.uint8)
    return background

def add_glasses(frame, glasses_png, landmarks_results):
    image_height, image_width = frame.shape[:2]
    if landmarks_results is not None:
        landmarks_list = landmarks_results.face_landmarks
        for idx in range(len(landmarks_list)):
            landmarks = landmarks_list[idx]

            left_eye = landmarks[33]
            right_eye = landmarks[263]
            x1 = int(left_eye.x * image_width)
            y1 = int(left_eye.y * image_height)
            x2 = int(right_eye.x * image_width)
            y2 = int(right_eye.y * image_height)

            center_x = (x1+x2) / 2
            center_y = (y1+y2) / 2
            dx = x2-x1
            dy = y2-y1
            eye_distance = (dx**2 + dy**2)**0.5
            sticker_width = int(eye_distance * 1.8)

            angle = -np.degrees(np.arctan2(dy, dx))
            scale = sticker_width / glasses_png.shape[1]
            sticker_height = int(glasses_png.shape[0] * scale)
            #input for cv.resize() needs to be int
            resized = cv.resize(glasses_png, (sticker_width, sticker_height))
            (h2, w2) = resized.shape[:2]
            center = (w2 // 2, h2 // 2)
            matrix = cv.getRotationMatrix2D(center, angle, 1.0)
            #get new dimensions of sticker after rotated
            cos = abs(matrix[0, 0])
            sin = abs(matrix[0, 1])
            new_w = int((h2*sin)+(w2*cos))
            new_h = int((h2*cos)+(w2*sin))
            matrix[0,2] += (new_w / 2) - center[0]
            matrix[1,2] += (new_h / 2) - center[1]

            rotated = cv.warpAffine(resized, matrix, (new_w, new_h))
            x = int(center_x - rotated.shape[1] / 2)
            y = int(center_y - rotated.shape[0] / 2)

            frame = overlay_sticker(frame, rotated, x, y)
    return frame

def add_hat(frame, hat_png, landmarks_results):
    image_h, image_w = frame.shape[:2]
    if landmarks_results is not None:
        landmarks_list = landmarks_results.face_landmarks
        for idx in range(len(landmarks_list)):
            landmarks = landmarks_list[idx]

            left_face = landmarks[234]
            right_face = landmarks[454]
            forehead = landmarks[10]

            x1 = int(left_face.x * image_w)
            y1 = int(left_face.y * image_h)
            x2 = int(right_face.x * image_w)
            y2 = int(right_face.y * image_h)
            forehead_x = int(forehead.x * image_w)
            forehead_y = int(forehead.y * image_h)
            center_x = (x2+x1) / 2
            dx = x2-x1
            dy = y2-y1
            head_distance = (dx**2 + dy**2)**0.5
            hat_width = int(head_distance * 2)

            angle = -np.degrees(np.arctan2(dy, dx))
            scale = hat_width / hat_png.shape[1]
            hat_height = int(scale * hat_png.shape[0])

            resized = cv.resize(hat_png, (hat_width, hat_height))
            h, w = resized.shape[:2]
            center = (w//2, h//2)
            matrix = cv.getRotationMatrix2D(center, angle, 1.0)

            cos = abs(matrix[0, 0])
            sin = abs(matrix[0, 1])
            new_w = int(w*cos + h*sin)
            new_h = int(w*sin + h*cos)

            matrix[0, 2] += new_w / 2 - center[0]
            matrix[1, 2] += new_h / 2 - center[1]

            rotated = cv.warpAffine(resized, matrix, (new_w, new_h))

            #get unit vector
            ux = dx / head_distance
            uy = dy / head_distance

            #get normal vector in direction of the head
            nx1, ny1 = -uy, ux
            nx2, ny2 = uy, -ux

            if  ny1 < ny2:
                nx, ny = nx1, ny1
            else:
                nx, ny = nx2, ny2
            
            offset = hat_height * 0.25
            hat_center_x = forehead_x + nx * offset
            hat_center_y = forehead_y + ny * offset

            x = int(hat_center_x - rotated.shape[1] / 2)
            y = int(hat_center_y - rotated.shape[0] / 2)
            frame = overlay_sticker(frame, rotated, x, y)

    return frame