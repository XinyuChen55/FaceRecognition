import os
import glob
import cv2
import json
from mmdet.apis import init_detector, inference_detector

TEST_IMG = "assets/test_imgs"
OUTPUT = "assets/outputs/mmdet_test"
VIS = os.path.join(OUTPUT, "vis")
BBOX = os.path.join(OUTPUT, "bboxes.json")
CONFIG = "third_party/mmdetection/configs/retinanet/retinanet_r50_fpn_1x_coco.py"
CHECKPOINT = "checkpoints/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth" 

def main():
    model = init_detector(CONFIG, CHECKPOINT)

    img_paths = glob.glob(os.path.join(TEST_IMG, "*.jpg"))
    all_bboxes ={}

    for p in img_paths:
        img = cv2.imread(p)
        result = inference_detector(model, img)

        img_copy = img.copy()
        boxes = result[0]
        keep = []
        for x1, y1, x2, y2, score in boxes:
            if score < 0.4:
                continue
            keep.append({"xyxy": [float(x1), float(y1), float(x2), float(y2)], "score": float(score)})
            
            x1,y1,x2,y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(img_copy, (x1, y1), (x2,y2), (0,255,0), 2)
        all_bboxes[os.path.basename(p)] = keep
        out = os.path.join(VIS, os.path.basename(p))
        cv2.imwrite(out, img_copy)
    with open(BBOX, "w", encoding="utf-8") as f:
        json.dump(all_bboxes, f, indent=2)

if __name__ == "__main__":
    main()