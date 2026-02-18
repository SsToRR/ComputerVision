import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("C:/Users/Miras/Desktop/dev/ComputerVision/image.png")
img_resized = cv2.resize(img, (256, 256))

gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

Gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
Gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

magnitude = cv2.magnitude(Gx, Gy)
magnitude_uint8 = cv2.convertScaleAbs(magnitude)

patch_mag = magnitude_uint8[130:135, 80:85]

print("Sobel magnitude 5x5 patch (rows 130-134, cols 80-84):\n", patch_mag)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(magnitude_uint8, cmap="gray")
plt.title("Sobel Gradient Magnitude")
plt.axis("off")

plt.tight_layout()
plt.show()
