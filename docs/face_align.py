from pathlib import Path
import cv2
import numpy as np
import json

def get_five_points(landmarks):
    left_eye = np.mean(landmarks[36:42], axis=0)
    right_eye = np.mean(landmarks[42:48], axis=0)
    nose = landmarks[30]
    left_mouth = landmarks[48]
    right_mouth = landmarks[54]

    five_pts = np.array([left_eye, right_eye, nose, left_mouth, right_mouth], dtype=np.float32)

    return five_pts

def align_face(img, landmarks, output_size):
    pts = get_five_points(landmarks)
    w, h = output_size
    dst = np.array([
        [0.34 * w, 0.38 * h],
        [0.66 * w, 0.38 * h],
        [0.50 * w, 0.56 * h],
        [0.38 * w, 0.75 * h],
        [0.62 * w, 0.75 * h]
    ], dtype = np.float32)

    Matrix, _ = cv2.estimateAffinePartial2D(pts, dst)
    results = cv2.warpAffine(img, Matrix, output_size)

    return results

JSON = Path('work_dirs/300w_hrnet/20260305_161323/test_vis')
OUT = Path('work_dirs/300w_hrnet/20260305_161323/aligned')

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    for json_path in JSON.glob("*_landmarks.json"):
        with open(json_path, "r") as f:
            data = json.load(f)
        keypoints = np.array(data["keypoints"][0], dtype=np.float32)
        img_path = data['img_path']
        img = cv2.imread(img_path)
        result_img = align_face(img, keypoints, (224,224))

        out_path = OUT / f"{Path(img_path).stem}_aligned{Path(img_path).suffix}"
        cv2.imwrite(str(out_path), result_img)

if __name__ == '__main__':
    main()
