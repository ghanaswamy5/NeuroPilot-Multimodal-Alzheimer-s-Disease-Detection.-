import os

print("Train AD:", len(os.listdir("data/MRI/train/AD")))
print("Train CN:", len(os.listdir("data/MRI/train/CN")))
print("Test AD:", len(os.listdir("data/MRI/test/AD")))
print("Test CN:", len(os.listdir("data/MRI/test/CN")))

import cv2
import matplotlib.pyplot as plt

img = cv2.imread("data/MRI/train/AD/OAS1_0053_MR1_mpr-1_125.jpg", 0)

plt.imshow(img, cmap='gray')
plt.show()