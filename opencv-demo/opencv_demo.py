import cv2
from pathlib import Path

in_path = Path("input.jpg")
out_path = Path("output_gray.jpg")

img = cv2.imread(str(in_path))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite(str(out_path), gray)
