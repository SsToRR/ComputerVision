import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.segmentation import slic, mark_boundaries
from skimage.color import label2rgb
from scipy import ndimage as ndi

# ===============================
# Load Image
# ===============================

image = cv2.imread("../image.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)
plt.title("Original Image")
plt.axis("off")
plt.show()

# ==========================================================
# TASK 1 — K-Means Segmentation in Color Space
# ==========================================================

pixels = image.reshape((-1, 3))
pixels = np.float32(pixels)

K_values = [2, 4, 6]

for K in K_values:
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    print(f"K = {K}")
    print("Cluster centers:")
    print(centers)
    print()

    segmented = centers[labels].reshape(image.shape)
    segmented = np.uint8(segmented)

    plt.imshow(segmented)
    plt.title(f"K-Means Segmentation (K={K})")
    plt.axis("off")
    plt.show()

# ==========================================================
# TASK 2 — HSV Color Space Threshold Segmentation
# ==========================================================

hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
hue = hsv[:, :, 0]

lower_hue = 30
upper_hue = 90

mask = np.zeros_like(hue, dtype=np.uint8)

for i in range(hue.shape[0]):
    for j in range(hue.shape[1]):
        if lower_hue <= hue[i, j] <= upper_hue:
            mask[i, j] = 255

pixel_count = np.sum(mask == 255)

print("Selected Hue range:", lower_hue, "to", upper_hue)
print("Number of segmented pixels:", int(pixel_count))

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(image)
plt.title("Original")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(hue, cmap="gray")
plt.title("Hue Channel")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(mask, cmap="gray")
plt.title("Segmentation Mask")
plt.axis("off")

plt.show()

# ==========================================================
# TASK 3 — Watershed Region-Based Segmentation
# ==========================================================

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

_, sure_fg = cv2.threshold(distance, 0.5 * distance.max(), 255, 0)
sure_fg = np.uint8(sure_fg)

sure_bg = cv2.dilate(binary, np.ones((3,3), np.uint8), iterations=3)
unknown = cv2.subtract(sure_bg, sure_fg)

num_markers, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

markers = cv2.watershed(image.copy(), markers)

segmented_ws = label2rgb(markers, image=image)

region_count = len(np.unique(markers)) - 1

print("Number of segmented regions (Watershed):", region_count)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(markers, cmap="jet")
plt.title("Markers")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(segmented_ws)
plt.title("Watershed Result")
plt.axis("off")

plt.show()

# ==========================================================
# TASK 4 — Superpixel Segmentation (SLIC)
# ==========================================================

segments = slic(image, n_segments=200, compactness=10, start_label=1)

superpixel_count = len(np.unique(segments))

print("Number of superpixels:", superpixel_count)

boundaries = mark_boundaries(image, segments)

plt.imshow(boundaries)
plt.title("SLIC Superpixels")
plt.axis("off")
plt.show()

# ==========================================================
# TASK 5 — Segmentation Result Comparison
# ==========================================================

# Select one region from K-means (for K=4)
kmeans_4 = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans_4.fit(pixels)

labels_4 = kmeans_4.labels_.reshape(image.shape[0], image.shape[1])

binary_kmeans = np.zeros_like(labels_4, dtype=np.uint8)

chosen_cluster = 0

for i in range(labels_4.shape[0]):
    for j in range(labels_4.shape[1]):
        if labels_4[i, j] == chosen_cluster:
            binary_kmeans[i, j] = 1

# Binary mask from HSV segmentation
binary_hsv = (mask == 255).astype(np.uint8)

# Manual IoU computation
intersection = np.sum((binary_kmeans == 1) & (binary_hsv == 1))
union = np.sum((binary_kmeans == 1) | (binary_hsv == 1))

iou = intersection / union if union != 0 else 0

print("Intersection:", int(intersection))
print("Union:", int(union))
print("IoU between K-means and HSV segmentation:", iou)

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(binary_kmeans, cmap="gray")
plt.title("Binary K-means Region")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(binary_hsv, cmap="gray")
plt.title("Binary HSV Region")
plt.axis("off")

plt.show()