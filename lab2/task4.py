import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("C:/Users/Miras/Desktop/dev/ComputerVision/image.png")
img_resized = cv2.resize(img, (256, 256))

median_bgr = cv2.medianBlur(img_resized, 5)

gray_original = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
gray_median = cv2.cvtColor(median_bgr, cv2.COLOR_BGR2GRAY)

patch_original = gray_original[130:135, 80:85]
patch_median = gray_median[130:135, 80:85]

print("Original grayscale 5x5 patch (rows 130-134, cols 80-84):\n", patch_original)
print("Median filtered grayscale 5x5 patch (rows 130-134, cols 80-84):\n", patch_median)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(gray_original, cmap="gray")
plt.title("Grayscale Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(gray_median, cmap="gray")
plt.title("Grayscale Median Filtered (k=5)")
plt.axis("off")

plt.tight_layout()
plt.show()
