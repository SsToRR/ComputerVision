import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load image (BGR)
img = cv2.imread("C:/Users/Miras/Desktop/dev/ComputerVision/image.png")
assert img is not None, "Image could not be loaded"

# Convert BGR -> RGB for correct visualization
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

plt.figure(figsize=(5,5))
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")
plt.show()

gray_norm = gray / 255.0

print("Grayscale image data type:", gray.dtype)
print("Normalized image data type:", gray_norm.dtype)
print("Min pixel value (normalized):", gray_norm.min())
print("Max pixel value (normalized):", gray_norm.max())

plt.figure(figsize=(6,4))
plt.hist(gray.ravel(), bins=256, range=(0, 255))
plt.title("Histogram of Grayscale Image (0–255)")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(6,4))
plt.hist(gray_norm.ravel(), bins=256, range=(0, 1))
plt.title("Histogram of Normalized Image (0–1)")
plt.xlabel("Normalized Pixel Intensity")
plt.ylabel("Frequency")
plt.show()

cv2.imwrite("grayscale.png", gray)

print("Task 3 completed. Grayscale image saved as 'grayscale.png'.")
