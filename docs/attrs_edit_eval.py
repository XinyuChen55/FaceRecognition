import lpips
loss_fn_alex = lpips.LPIPS(net='alex')

from torchvision import transforms
import os
from PIL import Image
import numpy as np
real_dir = "/home/chelsea/FaceRecognition/third_party/StarGAN/stargan_celeba/results/real"
fake_dir = "/home/chelsea/FaceRecognition/third_party/StarGAN/stargan_celeba/results/fake_smiling"

real_list = sorted(os.listdir(real_dir))
fake_list = sorted(os.listdir(fake_dir))
transform = transforms.ToTensor()
scores = []

for r, f in zip(real_list, fake_list):
    real_path = os.path.join(real_dir, r)
    fake_path = os.path.join(fake_dir, f)
    real_img = transform(Image.open(real_path)).unsqueeze(0)
    fake_img = transform(Image.open(fake_path)).unsqueeze(0)
    real_img = real_img * 2 - 1
    fake_img = fake_img * 2 - 1

    dist = loss_fn_alex(real_img, fake_img)
    scores.append(dist.item())

mean = np.mean(scores)
print(f"Average LPIPS: {mean:.4f}")