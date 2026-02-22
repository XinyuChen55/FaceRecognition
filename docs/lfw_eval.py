import os
import csv
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LFW_DIR = PROJECT_ROOT / "data" / "lfw" / "lfw-deepfunneled" /"lfw-deepfunneled"
PAIRS_CSV = PROJECT_ROOT / "data" / "lfw" / "pairs.csv"

OUT_DIR = PROJECT_ROOT / "assets" / "outputs" / "lfw_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_TXT = OUT_DIR / "lfw_eval_report.txt"

def lfw_img_path(lfw_root: Path, name: str, idx: int) -> Path:
    return lfw_root / name / f"{name}_{idx:04d}.jpg"


def pairs_csv(pairs_path: Path) -> List[Tuple[str, int, str, int, int]]:
    pairs = []

    with open(pairs_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    #rows = [row for row in rows if row]

    for row in rows[1:]:
        row = [c.strip() for c in row if c is not None and c.strip() != ""]
        if len(row) == 3:
            name, a, b = row
            pairs.append((name, int(a), name, int(b), 1))
        elif len(row) == 4:
            n1, a, n2, b = row
            pairs.append((n1, int(a), n2, int(b), 0))

    return pairs


def load_image_rgb(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


def build_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(), 
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def main():
    pairs = pairs_csv(PAIRS_CSV)

    model = InceptionResnetV1(pretrained="vggface2").eval().to("cuda:0")
    tfm = build_transform(160)

    emb_cache: Dict[str, np.ndarray] = {}

    @torch.no_grad()
    def get_embedding(img_path: Path) -> Optional[np.ndarray]:
        key = str(img_path)
        if key in emb_cache:
            return emb_cache[key]

        img = load_image_rgb(img_path)

        x = tfm(img).unsqueeze(0).to("cuda:0")
        emb = model(x) 
        emb = emb.squeeze(0).detach().cpu().numpy().astype(np.float32)

        emb_cache[key] = emb
        return emb

    dists = []
    labels = []
    missing_examples = []

    for i, (n1, idx1, n2, idx2, same_label) in enumerate(pairs):
        p1 = lfw_img_path(LFW_DIR, n1, idx1)
        p2 = lfw_img_path(LFW_DIR, n2, idx2)

        e1 = get_embedding(p1)
        e2 = get_embedding(p2)

        dist = float(np.linalg.norm(e1 - e2, ord=2))

        dists.append(dist)
        labels.append(int(same_label))

        if (i + 1) % 500 == 0:
            print(f"processed {i+1}/{len(pairs)} pairs")

    dists = np.array(dists, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    best_thr = None
    best_acc = -1.0

    for thr in np.arange(0.4, 2.0, 0.01):
        preds = (dists <= thr).astype(np.int32)
        acc = float((preds == labels).mean())
        if acc > best_acc:
            best_acc = acc
            best_thr = float(thr)

    same_dists = dists[labels == 1]
    diff_dists = dists[labels == 0]

    summary = {
        "num_pairs_total": int(len(pairs)),
        "best_threshold": float(best_thr),
        "best_accuracy": float(best_acc),
        "same_mean": float(same_dists.mean()),
        "same_std": float(same_dists.std()),
        "diff_mean": float(diff_dists.mean()),
        "diff_std": float(diff_dists.std()),
    }

    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()


