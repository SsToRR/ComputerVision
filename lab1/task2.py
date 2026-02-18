import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("C:/Users/Miras/Desktop/dev/ComputerVision/image.png")
assert img is not None, "Image could not be loaded"

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

resized = cv2.resize(img_rgb, (256, 256))

# Example crop (you can adjust values)
h, w, _ = img_rgb.shape

crop = img_rgb[
    int(h*0.4):int(h*0.8),   # vertical range
    int(w*0.2):int(w*0.7)    # horizontal range
]

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(img_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(resized)
plt.title("Resized (256×256)")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(crop)
plt.title("Cropped (ROI)")
plt.axis("off")

plt.show()
