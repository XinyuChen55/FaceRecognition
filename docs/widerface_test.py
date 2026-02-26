from pathlib import Path
from mmdet.apis import init_detector, inference_detector
import random

CONFIG = "configs/retinanet_r50_fpn_1x_widerface.py"
CHECKPOINT = "work_dirs/retinanet_r50_fpn_1x_widerface/latest.pth"
IMG_DIR = Path("data/widerface/WIDER_test/images")
OUT = Path("assets/outputs/widerface_test")

def main():
    model = init_detector(CONFIG, CHECKPOINT, device="cuda:0")

    imgs = [p for p in IMG_DIR.rglob("*.jpg")]
    random.seed(1)
    picks = random.sample(imgs, 10)

    for i, img_path in enumerate(picks, 1):
        result = inference_detector(model, str(img_path))
        name = img_path.relative_to(IMG_DIR)
        out_path = OUT / name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        model.show_result(
            str(img_path),
            result,
            score_thr=0.3,
            show=False,
            out_file=str(out_path),
        )

if __name__ == "__main__":
    main()