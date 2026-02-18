import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("C:/Users/Miras/Desktop/dev/ComputerVision/image.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (256, 256))

# Translation
dx, dy = 28, 34
M_translate = np.float32([
    [1, 0, dx],
    [0, 1, dy]
])

img_translated = cv2.warpAffine(
    img_resized,
    M_translate,
    (256, 256)
)

# 4. Rotation
center = (128, 128)  # center of the image
angle = 18
scale = 1.0

M_rotate = cv2.getRotationMatrix2D(center, angle, scale)

img_rotated = cv2.warpAffine(
    img_translated,
    M_rotate,
    (256, 256)
)

# 5. Display results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img_resized)
plt.title("Original (Resized)")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(img_translated)
plt.title("Translated")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(img_rotated)
plt.title("Translated + Rotated")
plt.axis("off")

plt.show()
