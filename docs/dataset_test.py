import os
import glob
import random
import cv2
import pandas as pd
import matplotlib.pyplot as plt

LFW_IMG = "data/lfw/lfw-deepfunneled/lfw-deepfunneled"
CELEBA_IMG = "data/celebA/img_align_celeba"
CELEBA_ATTR = "data/celebA_raw/list_attr_celeba.csv"
OUT = "assets/outputs/dataset_test"

random.seed(1)

def save_grid(img_paths, out_path, title="", n=16):
    chosen = random.sample(img_paths, min(n, len(img_paths)))
    cols = int(n ** 0.5)
    rows = (len(chosen) + cols -1) // cols

    plt.figure(figsize=(cols*2.5, rows*2.5))
    for i, p in enumerate(chosen, 1):
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(rows, cols, i)
        plt.imshow(img)
        plt.axis("off")
    
    if title:
        plt.suptitle(title)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def lfw_analysis():
    people = [name for name in glob.glob(os.path.join(LFW_IMG, "*"))]
    counts = {os.path.basename(d): len(glob.glob(os.path.join(d, "*.jpg"))) for d in people}
    
    total_imgs = sum(counts.values())
    total_people = len(counts)
    print(f"LFW contains number of images {total_imgs} and number of people {total_people}")

    top20 = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:20]
    names = [x[0] for x in top20]
    vals = [x[1] for x in top20]

    plt.figure(figsize=(10, 6))
    plt.barh(names[::-1], vals[::-1])
    plt.title("LFW Top 20 Image Count")
    plt.xlabel("Number of images")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "lfw_top20images.png"), dpi=200)
    plt.close()

    all_imgs = glob.glob(os.path.join(LFW_IMG, "*", "*.jpg"))
    save_grid(all_imgs, os.path.join(OUT, "lfw_random.png"), title="LFW: random samples")

def celeba_analysis():
    img_paths = glob.glob(os.path.join(CELEBA_IMG, "*.jpg"))
    print(f"CelebA contains number of images {len(img_paths)}")

    save_grid(img_paths, os.path.join(OUT, "celeba_random.png"), title="CelebA: random samples")

    df = pd.read_csv(CELEBA_ATTR) 
    attrs = ["Smiling", "Male", "Young", "Eyeglasses", "Wearing_Hat"]

    pos_counts = []
    neg_counts = []
    for a in attrs:
        col = df[a]
        pos = 0
        neg = 0
        for v in col:
            if v == 1:
                pos += 1
            else:
                neg += 1
        pos_counts.append(pos)
        neg_counts.append(neg)
    
    plt.figure(figsize=(10,6))
    plt.bar(attrs, pos_counts, label="Positive (1)")
    plt.bar(attrs, neg_counts, bottom=pos_counts, label="Negative (-1)")
    plt.ylabel("number of images")
    plt.title("CelebA: attribute distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "celeba_attribute_distribution.png"), dpi=200)
    plt.close()



lfw_analysis()
celeba_analysis()

