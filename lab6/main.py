import torch
import torchvision
import torchvision.transforms as T
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ==========================================================
# Load Image
# ==========================================================

image_bgr = cv2.imread("../cat.jpg")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")
plt.show()

# ==========================================================
# TASK 1 — Manual Bounding Box
# ==========================================================

# Manually define bounding box (x, y, width, height)
manual_x = 100
manual_y = 80
manual_w = 200
manual_h = 250

print("Manual Bounding Box (x, y, width, height):")
print((manual_x, manual_y, manual_w, manual_h))

image_with_box = image_rgb.copy()
cv2.rectangle(
    image_with_box,
    (manual_x, manual_y),
    (manual_x + manual_w, manual_y + manual_h),
    (255, 0, 0),
    3
)

plt.imshow(image_with_box)
plt.title("Manual Bounding Box")
plt.axis("off")
plt.show()

# ==========================================================
# TASK 2 — Pretrained Object Detection
# ==========================================================

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

transform = T.Compose([T.ToTensor()])
input_tensor = transform(image_rgb)

with torch.no_grad():
    outputs = model([input_tensor])

boxes = outputs[0]['boxes'].numpy()
scores = outputs[0]['scores'].numpy()
labels = outputs[0]['labels'].numpy()

confidence_threshold = 0.5

detected_indices = np.where(scores >= confidence_threshold)[0]

print("Number of detected objects (threshold=0.5):", len(detected_indices))

image_detected = image_rgb.copy()

for idx in detected_indices:
    x1, y1, x2, y2 = boxes[idx]
    score = scores[idx]
    cv2.rectangle(image_detected,
                  (int(x1), int(y1)),
                  (int(x2), int(y2)),
                  (0, 255, 0),
                  2)
    cv2.putText(image_detected,
                f"{score:.2f}",
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2)

plt.imshow(image_detected)
plt.title("Detected Objects (threshold=0.5)")
plt.axis("off")
plt.show()

# ==========================================================
# TASK 3 — Confidence Threshold Analysis
# ==========================================================

thresholds = [0.3, 0.5, 0.8]

for th in thresholds:
    indices = np.where(scores >= th)[0]
    print(f"Threshold {th} → Detected objects:", len(indices))
    
    img_temp = image_rgb.copy()
    for idx in indices:
        x1, y1, x2, y2 = boxes[idx]
        cv2.rectangle(img_temp,
                      (int(x1), int(y1)),
                      (int(x2), int(y2)),
                      (0, 255, 0),
                      2)
    
    plt.imshow(img_temp)
    plt.title(f"Detections (threshold={th})")
    plt.axis("off")
    plt.show()

# ==========================================================
# TASK 4 — Manual IoU Computation
# ==========================================================

if len(detected_indices) > 0:
    idx = detected_indices[0]
    dx1, dy1, dx2, dy2 = boxes[idx]

    manual_x1 = manual_x
    manual_y1 = manual_y
    manual_x2 = manual_x + manual_w
    manual_y2 = manual_y + manual_h

    inter_x1 = max(manual_x1, dx1)
    inter_y1 = max(manual_y1, dy1)
    inter_x2 = min(manual_x2, dx2)
    inter_y2 = min(manual_y2, dy2)

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height

    manual_area = manual_w * manual_h
    detected_area = (dx2 - dx1) * (dy2 - dy1)

    union = manual_area + detected_area - intersection

    iou = intersection / union if union != 0 else 0

    print("Intersection area:", intersection)
    print("Union area:", union)
    print("IoU:", iou)

else:
    print("No detection available for IoU computation.")

# ==========================================================
# TASK 5 — Detection Summary
# ==========================================================

print("Detection summary complete.")
print("Review number of detections at each threshold and IoU value.")