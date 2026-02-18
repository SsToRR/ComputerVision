import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("C:/Users/Miras/Desktop/dev/ComputerVision/image.png")
assert img is not None, "Image could not be loaded"

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Wrong (BGR shown as RGB)")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(img_rgb)
plt.title("Correct (RGB)")
plt.axis("off")

plt.show()

print("Image shape:", img.shape)
print("Number of channels:", img.shape[2])
print("Data type:", img.dtype)
