import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("C:/Users/Miras/Desktop/dev/ComputerVision/image.png")
assert img is not None, "Image could not be loaded"

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

kernel = np.ones((5,5), np.uint8)

erosion = cv2.erode(binary, kernel, iterations=1)
dilation = cv2.dilate(binary, kernel, iterations=1)

hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

# Define color range (green color segmentation)
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])

mask = cv2.inRange(hsv, lower_green, upper_green)
segmented = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

plt.figure(figsize=(14,8))

plt.subplot(2,3,1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2,3,2)
plt.imshow(binary, cmap="gray")
plt.title("Binary Threshold")
plt.axis("off")

plt.subplot(2,3,3)
plt.imshow(erosion, cmap="gray")
plt.title("Erosion")
plt.axis("off")

plt.subplot(2,3,4)
plt.imshow(dilation, cmap="gray")
plt.title("Dilation")
plt.axis("off")

plt.subplot(2,3,5)
plt.imshow(mask, cmap="gray")
plt.title("HSV Mask")
plt.axis("off")

plt.subplot(2,3,6)
plt.imshow(segmented)
plt.title("Color Segmentation Result")
plt.axis("off")

plt.show()