import os
import glob
import json
import numpy as np
import cv2
from mmhuman3d.apis.inference import init_model, inference_image_based_model

def xyxy_to_xywh(x1, y1, x2, y2):
    return float(x1), float(y1), float(x2-x1), float(y2-y1)

def draw_bbox(img, xyxy):
    x1,y1,x2,y2 = map(int, xyxy)
    cv2.rectangle(img, (x1,y1), (x2,y2), (0, 255,0), 2)

def project_to_image(pts, cam, bbox_xywh):
    pts = np.asarray(pts)
    cam = np.asarray(cam).reshape(-1)
    x, y, w, h = map(float, bbox_xywh)

    X = pts[:, 0]
    Y = pts[:, 1]

    s, tx, ty = float(cam[0]), float(cam[1]), float(cam[2])

    xn = s * X + tx
    yn = s * Y + ty

    px = x + (xn + 1.0) * 0.5 * w
    py = y + (yn + 1.0) * 0.5 * h

    return np.stack([px, py], axis=1)

def draw_points(img, pts, radius=2):
    for x, y in pts:
        cv2.circle(img, (int(x), int(y)), radius, (0, 0, 255), -1)

def main():
    img_dir = os.path.join("assets", "test_imgs")
    out_dir = os.path.join("assets", "outputs", "mmhuman_test", "vis")

    bbox_json = os.path.join("assets", "outputs", "mmdet_test", "bboxes.json")
    with open(bbox_json, "r") as f:
        bboxes = json.load(f)

    mmhuman_root = os.path.join("third_party", "mmhuman3d")
    mesh_cfg  = os.path.join(mmhuman_root, "configs/hmr/resnet50_hmr_pw3d.py")
    mesh_ckpt = os.path.join(mmhuman_root, "data/checkpoints/resnet50_hmr_pw3d.pth")

    mesh_model = init_model(mesh_cfg, mesh_ckpt, device="cuda:0")[0]

    for name, info in bboxes.items():
        p = os.path.join(img_dir, name)
        img = cv2.imread(p)

        best = max(info, key=lambda d: float(d.get("score", 0.0)))
        x1,y1,x2,y2 = best["xyxy"]
        score = float(best.get("score", 1.0))
        x,y,w,h = xyxy_to_xywh(x1,y1,x2,y2)

        det_results = [{"bbox": np.array([x,y,w,h,score])}]
        out = inference_image_based_model(mesh_model, img, det_results, bbox_thr=0.0, format="xywh")

        vis = img.copy()
        draw_bbox(vis, (x1,y1,x2,y2))

        od = out[0]
        pts = project_to_image(od["keypoints_3d"], od["camera"], [x,y,w,h])
        draw_points(vis, pts, radius=2)

        cv2.imwrite(os.path.join(out_dir, name), vis)

if __name__ == "__main__":
    main()