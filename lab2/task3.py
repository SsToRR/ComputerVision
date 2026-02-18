import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read and resize image
img = cv2.imread("C:/Users/Miras/Desktop/dev/ComputerVision/image.png")
img_resized = cv2.resize(img, (256, 256))

# Translation matrix
dx, dy = 28, 34
M_translate = np.float32([[1, 0, dx],[0, 1, dy]])

# Rotation matrix
center = (128, 128)
M_rotate = cv2.getRotationMatrix2D(center, 18, 1.0)

# Apply transformations
translated_img = cv2.warpAffine(img_resized, M_translate, (256, 256))
rotated_img = cv2.warpAffine(translated_img, M_rotate, (256, 256))

# Points to transform
points = np.float32([[15, 210],[128, 128],[210, 35]]).reshape(-1, 1, 2)

# Transform points
translated_points = cv2.transform(points, M_translate)
rotated_points = cv2.transform(translated_points, M_rotate)

# Print matrices and points
print("Translation matrix:\n", M_translate)
print("Rotation matrix:\n", M_rotate)

print("Original points:\n", points.reshape(-1, 2))
print("After translation:\n", translated_points.reshape(-1, 2))
print("After rotation:\n", rotated_points.reshape(-1, 2))

# Draw points on images
img_points = img_resized.copy()
transformed_points_img = rotated_img.copy()

for p in points.reshape(-1, 2):
    cv2.circle(img_points, tuple(p.astype(int)), 4, (0, 0, 255), -1)

for p in rotated_points.reshape(-1, 2):
    cv2.circle(transformed_points_img, tuple(p.astype(int)), 4, (0, 255, 0), -1)

# Display
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1); plt.imshow(cv2.cvtColor(img_points, cv2.COLOR_BGR2RGB)); plt.axis("off")
plt.subplot(1, 2, 2); plt.imshow(cv2.cvtColor(transformed_points_img, cv2.COLOR_BGR2RGB)); plt.axis("off")
plt.show()
