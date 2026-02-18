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

pool_size = 2
stride = 2

h, w = magnitude_uint8.shape
out_h, out_w = h // stride, w // stride
pooled = np.zeros((out_h, out_w), dtype=np.uint8)

for y in range(0, h, stride):
    for x in range(0, w, stride):
        window = magnitude_uint8[y:y+pool_size, x:x+pool_size]
        pooled[y//stride, x//stride] = np.max(window)

print("Input shape:", magnitude_uint8.shape)
print("Output shape:", pooled.shape)

input_block = magnitude_uint8[130:134, 80:84]
pooled_block = pooled[65:67, 40:42]

print("Input 4x4 block (rows 130-133, cols 80-83):\n", input_block)
print("Pooled 2x2 block (rows 65-66, cols 40-41):\n", pooled_block)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(magnitude_uint8, cmap="gray")
plt.title("Gradient Magnitude (Input)")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(pooled, cmap="gray")
plt.title("Maxpooled Output (2x2, stride 2)")
plt.axis("off")

plt.tight_layout()
plt.show()
