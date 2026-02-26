import json
from pathlib import Path
import cv2

def parse_wider_txt(txt_path: Path):
    lines = txt_path.read_text().strip().splitlines()

    i = 0
    while i < len(lines):
        img_path = lines[i].strip()
        i += 1
        n = int(lines[i].strip())
        i += 1
        boxes = []
        for j in range(n):
            parts = lines[i].strip().split()
            i += 1
            x, y, w, h = map(float, parts[:4])
            if int(parts[7]) == 1:
                continue
            boxes.append([x, y, w, h])
        if n == 0:
            i += 1
        yield img_path, boxes

def wider_to_coco(img_root: Path, txt_path: Path, out: Path):
    images = []
    annotations = []
    ann_id = 1
    img_id = 1

    for img_name, boxes in parse_wider_txt(txt_path):
        img_path = img_root / img_name
        img = cv2.imread(img_path)
        height = img.shape[0]
        width = img.shape[1]

        images.append({
            "id": img_id,
            "file_name": img_name,
            "width": width,
            "height": height
        })

        for (x, y, w, h) in boxes:
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [x, y, w, h],
                "area": float(w * h),
                "iscrowd": 0
            })
            ann_id += 1
        img_id += 1
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{'id': 1, "name": "face"}]
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(coco))

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data" / "widerface"

    train_root = data_root / "WIDER_train" / "images"
    val_root = data_root / "WIDER_val" / "images"
    split_root = data_root /"wider_face_split"

    OUT = data_root / "annotations"
    wider_to_coco(train_root, split_root / "wider_face_train_bbx_gt.txt", OUT / "train_coco.json")
    wider_to_coco(val_root, split_root / "wider_face_val_bbx_gt.txt", OUT / "val_coco.json")
