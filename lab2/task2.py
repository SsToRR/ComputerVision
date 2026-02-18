import cv2
import numpy as np

# Read and resize image
img = cv2.imread("C:/Users/Miras/Desktop/dev/ComputerVision/image.png")
img_resized = cv2.resize(img, (256, 256))

# Translation
dx, dy = 28, 34
M_translate = np.float32([[1, 0, dx],[0, 1, dy]])
translated_img = cv2.warpAffine(img_resized, M_translate, (256, 256))

# Rotation
center = (128, 128)
M_rotate = cv2.getRotationMatrix2D(center, 18, 1.0)
final_img = cv2.warpAffine(translated_img, M_rotate, (256, 256))

# Pixel coordinates
x, y = 155, 60

# Pixel values
original_pixel = img_resized[y, x]
final_pixel = final_img[y, x]

print("Original image pixel at (155, 60):", original_pixel)
print("Final transformed image pixel at (155, 60):", final_pixel)