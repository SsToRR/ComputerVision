import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("C:/Users/Miras/Desktop/dev/ComputerVision/image.png")
img_resized = cv2.resize(img, (256, 256))
gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

sobel_x_kernel = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=np.float32)

padded = np.pad(gray, pad_width=1, mode='constant', constant_values=0)

manual_sobel = np.zeros_like(gray, dtype=np.float32)

for y in range(256):
    for x in range(256):
        region = padded[y:y+3, x:x+3]
        manual_sobel[y, x] = np.sum(region * sobel_x_kernel)

opencv_sobel = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)

manual_uint8 = cv2.convertScaleAbs(manual_sobel)
opencv_uint8 = cv2.convertScaleAbs(opencv_sobel)

patch_manual = manual_uint8[130:135, 80:85]
patch_opencv = opencv_uint8[130:135, 80:85]

print("Manual Sobel X 5x5 patch:\n", patch_manual)
print("OpenCV Sobel X 5x5 patch:\n", patch_opencv)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(manual_uint8, cmap="gray")
plt.title("Manual Sobel X")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(opencv_uint8, cmap="gray")
plt.title("OpenCV Sobel X")
plt.axis("off")

plt.tight_layout()
plt.show()
