import cv2 as cv
import numpy as np

def create_face_mask(frame, landmarks):
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    face_outline_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 
                    152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    
    face_outline_list = []
    for i in face_outline_indices:
        x = int(landmarks[i].x * w)
        y = int(landmarks[i].y * h)
        face_outline_list += [(x, y)]
    face_outline = np.array(face_outline_list, dtype=np.int32)

    cv.fillPoly(mask, [face_outline], 255)
    mask = cv.GaussianBlur(mask, (15, 15), 0)

    return mask

def apply_skin_smooth(frame, landmarks_results, strength):
    landmarks_list = landmarks_results.face_landmarks
    for landmarks in landmarks_list:
        mask = create_face_mask(frame, landmarks)
        smooth = cv.bilateralFilter(frame, 9, 75, 75)

        alpha = (mask / 255.0 * strength)[:, :, None]   #add one more dimension
        frame = (alpha * smooth + (1- alpha) * frame).astype(np.uint8)

    return frame

def apply_whitening(frame, landmarks_results, strength):
    landmarks_list = landmarks_results.face_landmarks
    for landmarks in landmarks_list:
        mask = create_face_mask(frame, landmarks)
        whitening = cv.convertScaleAbs(frame, alpha=1.0, beta=strength)
        alpha = (mask / 255.0)[:, :, None]
        frame = (alpha*whitening + (1-alpha)*frame).astype(np.uint8)

    return frame

def create_lip_mask(frame, landmarks):
    h, w = frame.shape[:2]
    mask_u = np.zeros((h, w), dtype=np.uint8)
    mask_l = np.zeros((h, w), dtype=np.uint8)

    upper_indices = [61, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 
                     291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
    lower_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 
                     291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]

    upper_outline = []
    for i in upper_indices:
        x = landmarks[i].x * w
        y = landmarks[i].y * h
        upper_outline += [(x, y)]
    upper_outline = np.array(upper_outline, dtype=np.int32)

    lower_outline = []
    for i in lower_indices:
        x = landmarks[i].x * w
        y = landmarks[i].y * h
        lower_outline += [(x, y)]
    lower_outline = np.array(lower_outline, dtype=np.int32)

    cv.fillPoly(mask_u, [upper_outline], 255)
    cv.fillPoly(mask_l, [lower_outline], 255)
    mask_u = cv.GaussianBlur(mask_u, (5, 5), 0)
    mask_l = cv.GaussianBlur(mask_l, (5, 5), 0)

    mask = cv.add(mask_u, mask_l)
    return mask

def apply_lipstick(frame, landmarks_results, color, strength):
    landmarks_list = landmarks_results.face_landmarks
    for landmarks in landmarks_list:
        mask = create_lip_mask(frame, landmarks)
        color_layer = np.zeros_like(frame, dtype=np.uint8)
        color_layer[:] = color

        alpha = (mask / 255.0 * strength)[:, :, None]
        frame = (alpha*color_layer + (1-alpha)*frame).astype(np.uint8)

    return frame