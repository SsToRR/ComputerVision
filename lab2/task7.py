import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("C:/Users/Miras/Desktop/dev/ComputerVision/image.png")
img_resized = cv2.resize(img, (256, 256))
gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

sobel_x_kernel = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=np.float32)

padded_zero = np.pad(gray, pad_width=1, mode='constant', constant_values=0)
padded_reflect = np.pad(gray, pad_width=1, mode='reflect')

sobel_zero = np.zeros_like(gray, dtype=np.float32)
sobel_reflect = np.zeros_like(gray, dtype=np.float32)

for y in range(256):
    for x in range(256):
        region_zero = padded_zero[y:y+3, x:x+3]
        region_reflect = padded_reflect[y:y+3, x:x+3]
        sobel_zero[y, x] = np.sum(region_zero * sobel_x_kernel)
        sobel_reflect[y, x] = np.sum(region_reflect * sobel_x_kernel)

sobel_zero_uint8 = cv2.convertScaleAbs(sobel_zero)
sobel_reflect_uint8 = cv2.convertScaleAbs(sobel_reflect)

patch_zero = sobel_zero_uint8[0:5, 0:5]
patch_reflect = sobel_reflect_uint8[0:5, 0:5]

print("Zero padding Sobel X 5x5 patch (rows 0-4, cols 0-4):\n", patch_zero)
print("Reflect padding Sobel X 5x5 patch (rows 0-4, cols 0-4):\n", patch_reflect)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(sobel_zero_uint8, cmap="gray")
plt.title("Manual Sobel X (Zero Padding)")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(sobel_reflect_uint8, cmap="gray")
plt.title("Manual Sobel X (Reflect Padding)")
plt.axis("off")

plt.tight_layout()
plt.show()
