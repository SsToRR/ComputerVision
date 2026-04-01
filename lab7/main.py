import os
import time
import math
import copy
import random
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.utils import draw_bounding_boxes
from torchvision.ops import box_iou, nms


# ============================================================
# LAB 7. Object Detection with CNN
# Dataset: Face Mask Detection (Kaggle, Pascal VOC XML format)
# Model: Faster R-CNN ResNet50 FPN
# ============================================================
# IMPORTANT:
# 1. Download dataset and unzip it.
# 2. Change DATASET_ROOT below.
# 3. This code assumes Pascal VOC XML annotations.
# 4. It uses xmin, ymin, xmax, ymax.
# ============================================================


# =========================
# GLOBAL SETTINGS
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

DATASET_ROOT = "./"  # 
IMAGES_DIR = os.path.join(DATASET_ROOT, "images")
ANNOTATIONS_DIR = os.path.join(DATASET_ROOT, "annotations")

# Fixed subset to reduce training time
MAX_IMAGES_TO_USE = 300
TRAIN_RATIO = 0.8
BATCH_SIZE = 4
NUM_WORKERS = 0

# Classes
CLASS_NAMES = ["background", "with_mask", "without_mask", "mask_weared_incorrect"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

# Training params from lab
BASELINE_EPOCHS = 5
FINETUNE_EPOCHS = 5
BASELINE_LR = 0.005
FINETUNE_LR = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

CONF_THRESH_DEFAULT = 0.5
IOU_MATCH_THRESHOLD = 0.5


# =========================
# UTILS
# =========================
# Sets random seeds for reproducible results across libraries.
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Collects and sorts all Pascal VOC annotation files.
def get_all_xml_files(annotations_dir):
    files = [f for f in os.listdir(annotations_dir) if f.endswith(".xml")]
    files.sort()
    return files


# Parses one Pascal VOC XML file into image metadata and boxes.
def parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find("filename").text

    boxes = []
    labels = []
    difficult = []

    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name not in CLASS_TO_IDX:
            continue

        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(CLASS_TO_IDX[name])

        diff_tag = obj.find("difficult")
        difficult.append(int(diff_tag.text) if diff_tag is not None else 0)

    return {
        "filename": filename,
        "width": width,
        "height": height,
        "boxes": boxes,
        "labels": labels,
        "difficult": difficult,
    }


# Draws bounding boxes and optional labels on an image.
def show_image_with_boxes(image, boxes, labels=None, scores=None, title="Image"):
    if isinstance(image, Image.Image):
        image_tensor = transforms.ToTensor()(image)
    else:
        image_tensor = image.detach().cpu()

    image_uint8 = (image_tensor * 255).to(torch.uint8)

    text_labels = []
    if labels is not None:
        for i, lab in enumerate(labels):
            class_name = CLASS_NAMES[int(lab)] if int(lab) < len(CLASS_NAMES) else str(lab)
            if scores is not None:
                text_labels.append(f"{class_name}: {scores[i]:.2f}")
            else:
                text_labels.append(class_name)

    if len(boxes) > 0:
        drawn = draw_bounding_boxes(image_uint8, boxes=torch.tensor(boxes, dtype=torch.float32), labels=text_labels, width=2)
    else:
        drawn = image_uint8

    plt.figure(figsize=(8, 6))
    plt.imshow(drawn.permute(1, 2, 0))
    plt.title(title)
    plt.axis("off")
    plt.show()


# Counts labeled object instances across annotation files.
def count_instances(xml_files):
    counter = Counter()
    for xml_name in xml_files:
        ann = parse_voc_xml(os.path.join(ANNOTATIONS_DIR, xml_name))
        for lab in ann["labels"]:
            counter[CLASS_NAMES[lab]] += 1
    return counter


# Packs detection samples into list-based batches for DataLoader.
def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


# Builds the image transform pipeline used by the dataset.
def get_transform(train=False):
    transform_list = []
    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)


# Counts model parameters that will be updated during training.
def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Freezes backbone layers so only detection heads are trained.
def freeze_backbone(model):
    for p in model.backbone.parameters():
        p.requires_grad = False


# Unfreezes only the last ResNet backbone block for fine-tuning.
def unfreeze_last_backbone_block(model):
    # First freeze everything in backbone
    for p in model.backbone.parameters():
        p.requires_grad = False

    # Then unfreeze last block from ResNet body (layer4)
    if hasattr(model.backbone, "body") and hasattr(model.backbone.body, "layer4"):
        for p in model.backbone.body.layer4.parameters():
            p.requires_grad = True


# Runs one training epoch and returns the average loss.
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    return total_loss / max(len(loader), 1)


# Runs object detection inference and moves outputs to CPU.
def predict(model, images, device):
    model.eval()
    with torch.no_grad():
        images = [img.to(device) for img in images]
        outputs = model(images)
    return [{k: v.detach().cpu() for k, v in out.items()} for out in outputs]


# Keeps only detections above the chosen confidence threshold.
def filter_predictions(output, conf_thresh=0.5):
    boxes = output["boxes"]
    labels = output["labels"]
    scores = output["scores"]

    keep = scores >= conf_thresh
    return {
        "boxes": boxes[keep],
        "labels": labels[keep],
        "scores": scores[keep],
    }


# Measures average inference time per image on a data loader.
def evaluate_inference_speed(model, loader, device):
    model.eval()
    times = []
    with torch.no_grad():
        for images, _ in loader:
            for img in images:
                img = img.to(device)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.time()
                _ = model([img])
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end = time.time()
                times.append(end - start)
    return float(np.mean(times)) if times else 0.0


# Matches predicted boxes to ground truth boxes using IoU.
def compute_matches_for_image(pred_boxes, gt_boxes, iou_thresh=0.5):
    if len(pred_boxes) == 0:
        iou_matrix = torch.zeros((0, len(gt_boxes)))
        return iou_matrix, [], 0, 0, len(gt_boxes)
    if len(gt_boxes) == 0:
        iou_matrix = torch.zeros((len(pred_boxes), 0))
        return iou_matrix, [], 0, len(pred_boxes), 0

    iou_matrix = box_iou(pred_boxes, gt_boxes)
    matched_pairs = []
    used_gt = set()
    used_pred = set()

    flat = []
    for i in range(iou_matrix.shape[0]):
        for j in range(iou_matrix.shape[1]):
            flat.append((iou_matrix[i, j].item(), i, j))
    flat.sort(reverse=True, key=lambda x: x[0])

    for iou, i, j in flat:
        if iou < iou_thresh:
            continue
        if i not in used_pred and j not in used_gt:
            matched_pairs.append((i, j, iou))
            used_pred.add(i)
            used_gt.add(j)

    tp = len(matched_pairs)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return iou_matrix, matched_pairs, tp, fp, fn


# Computes precision-recall points and AP for one class.
def compute_ap_for_class(model, loader, class_id, device, iou_thresh=0.5, conf_thresh=0.0):
    model.eval()
    all_predictions = []
    total_gts = 0

    with torch.no_grad():
        for images, targets in loader:
            images_dev = [img.to(device) for img in images]
            outputs = model(images_dev)

            for out, target in zip(outputs, targets):
                pred_boxes = out["boxes"].detach().cpu()
                pred_scores = out["scores"].detach().cpu()
                pred_labels = out["labels"].detach().cpu()

                gt_boxes = target["boxes"].detach().cpu()
                gt_labels = target["labels"].detach().cpu()

                pred_mask = (pred_labels == class_id) & (pred_scores >= conf_thresh)
                gt_mask = gt_labels == class_id

                p_boxes = pred_boxes[pred_mask]
                p_scores = pred_scores[pred_mask]
                g_boxes = gt_boxes[gt_mask]

                total_gts += len(g_boxes)

                matched_gt = set()
                if len(p_boxes) > 0 and len(g_boxes) > 0:
                    ious = box_iou(p_boxes, g_boxes)
                else:
                    ious = torch.zeros((len(p_boxes), len(g_boxes)))

                order = torch.argsort(p_scores, descending=True)
                p_boxes = p_boxes[order]
                p_scores = p_scores[order]
                if ious.numel() > 0:
                    ious = ious[order]

                for i in range(len(p_boxes)):
                    best_iou = 0.0
                    best_j = -1
                    for j in range(len(g_boxes)):
                        if j in matched_gt:
                            continue
                        cur_iou = ious[i, j].item() if ious.numel() > 0 else 0.0
                        if cur_iou > best_iou:
                            best_iou = cur_iou
                            best_j = j

                    if best_iou >= iou_thresh and best_j != -1:
                        matched_gt.add(best_j)
                        all_predictions.append((p_scores[i].item(), 1))
                    else:
                        all_predictions.append((p_scores[i].item(), 0))

    if total_gts == 0:
        return [0.0], [0.0], 0.0

    all_predictions.sort(key=lambda x: x[0], reverse=True)

    tp_cum = 0
    fp_cum = 0
    precisions = []
    recalls = []

    for _, is_tp in all_predictions:
        if is_tp:
            tp_cum += 1
        else:
            fp_cum += 1
        precision = tp_cum / (tp_cum + fp_cum + 1e-8)
        recall = tp_cum / (total_gts + 1e-8)
        precisions.append(precision)
        recalls.append(recall)

    recalls = np.array(recalls)
    precisions = np.array(precisions)

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])

    return recalls.tolist(), precisions.tolist(), float(ap)


# Computes per-class AP values and the mean average precision.
def compute_map(model, loader, device, iou_thresh=0.5):
    ap_per_class = {}
    valid_aps = []

    for class_id in range(1, NUM_CLASSES):
        recalls, precisions, ap = compute_ap_for_class(model, loader, class_id, device, iou_thresh=iou_thresh)
        ap_per_class[CLASS_NAMES[class_id]] = ap
        valid_aps.append(ap)

    mAP = float(np.mean(valid_aps)) if valid_aps else 0.0
    return ap_per_class, mAP


# Plots baseline and fine-tuning loss curves for comparison.
def plot_losses(baseline_losses, finetune_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(baseline_losses) + 1), baseline_losses, marker='o', label='Baseline (frozen backbone)')
    plt.plot(range(1, len(finetune_losses) + 1), finetune_losses, marker='s', label='Fine-tuning (layer4 unfrozen)')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.show()


# Visualizes predictions for selected dataset sample indices.
def visualize_predictions_on_indices(model, dataset, indices, device, conf_thresh=0.5, title_prefix=""):
    model.eval()
    for idx in indices:
        image, target = dataset[idx]
        output = predict(model, [image], device)[0]
        output = filter_predictions(output, conf_thresh=conf_thresh)

        show_image_with_boxes(
            image=image,
            boxes=output["boxes"].tolist(),
            labels=output["labels"].tolist(),
            scores=output["scores"].tolist(),
            title=f"{title_prefix} Sample {idx}"
        )


# Shows ground-truth and predicted boxes together on one image.
def visualize_gt_and_pred(image, gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, title=""):
    img_tensor = (image * 255).to(torch.uint8)

    gt_texts = [f"GT:{CLASS_NAMES[int(x)]}" for x in gt_labels]
    pred_texts = [f"PR:{CLASS_NAMES[int(pred_labels[i])]}:{pred_scores[i]:.2f}" for i in range(len(pred_labels))]

    if len(gt_boxes) > 0:
        img_tensor = draw_bounding_boxes(img_tensor, gt_boxes, labels=gt_texts, width=3)
    if len(pred_boxes) > 0:
        img_tensor = draw_bounding_boxes(img_tensor, pred_boxes, labels=pred_texts, width=2)

    plt.figure(figsize=(8, 6))
    plt.imshow(img_tensor.permute(1, 2, 0))
    plt.title(title)
    plt.axis("off")
    plt.show()


# =========================
# TASK 2. DATASET CLASS
# =========================
class FaceMaskDetectionDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, xml_files, transforms=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.xml_files = xml_files
        self.transforms = transforms

    def __len__(self):
        return len(self.xml_files)

    def __getitem__(self, idx):
        xml_name = self.xml_files[idx]
        ann = parse_voc_xml(os.path.join(self.annotations_dir, xml_name))

        img_path = os.path.join(self.images_dir, ann["filename"])
        image = Image.open(img_path).convert("RGB")

        boxes = torch.tensor(ann["boxes"], dtype=torch.float32)
        labels = torch.tensor(ann["labels"], dtype=torch.int64)

        if boxes.numel() == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if len(boxes) > 0 else torch.zeros((0,), dtype=torch.float32)
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            image = self.transforms(image)

        return image, target


# =========================
# TASK 3. MODEL INIT
# =========================
# Creates a Faster R-CNN model with a custom class head.
def get_faster_rcnn_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# =========================
# TASK 4 and 15. FEATURE MAPS
# =========================
# Extracts backbone feature pyramid maps for one image.
def extract_backbone_features(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        features = model.backbone(image_tensor.unsqueeze(0))
    return {k: v.detach().cpu() for k, v in features.items()}


# Displays a few channels from backbone feature maps.
def visualize_feature_maps(feature_dict, max_maps_per_level=2, title_prefix="Feature"):
    shown = 0
    plt.figure(figsize=(14, 10))
    plot_idx = 1

    for level_name, fmap in feature_dict.items():
        channels = fmap.shape[1]
        for c in range(min(max_maps_per_level, channels)):
            plt.subplot(3, 2, plot_idx)
            plt.imshow(fmap[0, c].numpy(), cmap='viridis')
            plt.title(f"{title_prefix} {level_name} channel {c}")
            plt.axis('off')
            plot_idx += 1
            shown += 1
            if shown >= 6:
                plt.tight_layout()
                plt.show()
                return

    plt.tight_layout()
    plt.show()


# Compares three backbone feature levels side by side.
def compare_three_backbone_levels(feature_dict):
    keys = list(feature_dict.keys())[:3]
    if len(keys) < 3:
        print("Not enough backbone levels to compare.")
        return

    plt.figure(figsize=(15, 4))
    for i, k in enumerate(keys, start=1):
        fmap = feature_dict[k]
        plt.subplot(1, 3, i)
        plt.imshow(fmap[0, 0].numpy(), cmap='viridis')
        plt.title(f"{k} | shape={tuple(fmap.shape)}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# =========================
# TASK 11. NMS ANALYSIS
# =========================
# Applies confidence filtering and per-class non-maximum suppression.
def run_custom_nms_analysis(output, conf_thresh=0.05, nms_thresh=0.5):
    boxes = output["boxes"]
    scores = output["scores"]
    labels = output["labels"]

    conf_keep = scores >= conf_thresh
    boxes = boxes[conf_keep]
    scores = scores[conf_keep]
    labels = labels[conf_keep]

    before_nms = len(boxes)

    final_keep = []
    for class_id in labels.unique().tolist() if len(labels) > 0 else []:
        cls_mask = labels == class_id
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        cls_indices = torch.where(cls_mask)[0]
        keep = nms(cls_boxes, cls_scores, nms_thresh)
        final_keep.extend(cls_indices[keep].tolist())

    final_keep = sorted(final_keep)
    after_nms = len(final_keep)

    return {
        "before_nms": before_nms,
        "after_nms": after_nms,
        "boxes": boxes[final_keep] if len(final_keep) > 0 else torch.zeros((0, 4)),
        "scores": scores[final_keep] if len(final_keep) > 0 else torch.zeros((0,)),
        "labels": labels[final_keep] if len(final_keep) > 0 else torch.zeros((0,), dtype=torch.int64),
    }


# =========================
# TASK 14. ANCHOR ANALYSIS
# =========================
# Prints anchor sizes and aspect ratios used by the RPN.
def inspect_anchor_generator(model):
    ag = model.rpn.anchor_generator
    sizes = ag.sizes
    aspect_ratios = ag.aspect_ratios

    print("Anchor sizes (scales) per feature map level:")
    print(sizes)
    print("Aspect ratios per feature map level:")
    print(aspect_ratios)
    print("Number of feature levels:", len(sizes))

    for i, (s, ar) in enumerate(zip(sizes, aspect_ratios)):
        print(f"Level {i}: {len(s)} scales, {len(ar)} aspect ratios")


# =========================
# MAIN PIPELINE
# =========================
# Runs the full dataset, training, evaluation, and visualization pipeline.
def main():
    set_seed(SEED)

    # --------------------------------------------------
    # TASK 1. Dataset selection with annotation format
    # --------------------------------------------------
    xml_files = get_all_xml_files(ANNOTATIONS_DIR)
    print("Total annotation files found:", len(xml_files))

    if MAX_IMAGES_TO_USE is not None:
        xml_files = xml_files[:MAX_IMAGES_TO_USE]

    print("Subset size used:", len(xml_files))
    print("Classes:", CLASS_NAMES[1:])
    print("Annotation format: xmin, ymin, xmax, ymax")

    class_counter = count_instances(xml_files)
    print("Object instances per class:")
    for cls_name in CLASS_NAMES[1:]:
        print(f"{cls_name}: {class_counter[cls_name]}")

    plt.figure(figsize=(7, 4))
    plt.bar(list(class_counter.keys()), list(class_counter.values()))
    plt.title("Number of Object Instances per Class")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()

    # Show at least two images with GT boxes
    sample_xmls = xml_files[:2]
    for xml_name in sample_xmls:
        ann = parse_voc_xml(os.path.join(ANNOTATIONS_DIR, xml_name))
        img = Image.open(os.path.join(IMAGES_DIR, ann["filename"])).convert("RGB")
        show_image_with_boxes(
            img,
            ann["boxes"],
            ann["labels"],
            title=f"Ground Truth: {ann['filename']}"
        )

    # --------------------------------------------------
    # TASK 2. Detection DataLoader
    # --------------------------------------------------
    full_dataset = FaceMaskDetectionDataset(
        images_dir=IMAGES_DIR,
        annotations_dir=ANNOTATIONS_DIR,
        xml_files=xml_files,
        transforms=get_transform(train=True)
    )

    train_size = int(TRAIN_RATIO * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    batch_images, batch_targets = next(iter(train_loader))
    print("\nOne batch structure:")
    print("Number of images in batch:", len(batch_images))
    print("Image tensor shape of first image:", batch_images[0].shape)
    print("Target keys:", batch_targets[0].keys())
    for k, v in batch_targets[0].items():
        if torch.is_tensor(v):
            print(f"{k}: shape={tuple(v.shape)}, dtype={v.dtype}")

    # --------------------------------------------------
    # TASK 3. Pretrained CNN detector initialization
    # --------------------------------------------------
    model = get_faster_rcnn_model(NUM_CLASSES)
    model.to(DEVICE)

    print("\nModel architecture:\n")
    print(model)
    print("\nNumber of trainable parameters:", count_trainable_params(model))
    print("Backbone role: extracts visual features from image.")
    print("Detection head role: classifies regions and regresses bounding boxes.")

    # --------------------------------------------------
    # TASK 4. CNN feature map visualization
    # --------------------------------------------------
    sample_img, _ = full_dataset[0]
    feature_dict = extract_backbone_features(model, sample_img, DEVICE)
    print("\nBackbone feature levels:")
    for k, v in feature_dict.items():
        print(k, tuple(v.shape))

    visualize_feature_maps(feature_dict, max_maps_per_level=2, title_prefix="Task4")

    # --------------------------------------------------
    # TASK 5. Baseline training with frozen backbone
    # --------------------------------------------------
    freeze_backbone(model)
    print("\nTrainable parameters after freezing backbone:", count_trainable_params(model))

    baseline_optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=BASELINE_LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )

    baseline_losses = []
    for epoch in range(BASELINE_EPOCHS):
        epoch_loss = train_one_epoch(model, train_loader, baseline_optimizer, DEVICE)
        baseline_losses.append(epoch_loss)
        print(f"Baseline Epoch [{epoch+1}/{BASELINE_EPOCHS}] Loss: {epoch_loss:.4f}")

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, BASELINE_EPOCHS + 1), baseline_losses, marker='o')
    plt.title('Baseline Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

    baseline_model = copy.deepcopy(model)

    # Run inference on 5 test images
    test_indices_global = list(range(min(5, len(test_dataset))))
    print("\nBaseline predictions on 5 test images:")
    visualize_predictions_on_indices(baseline_model, test_dataset, test_indices_global, DEVICE, conf_thresh=CONF_THRESH_DEFAULT, title_prefix="Baseline")

    # --------------------------------------------------
    # TASK 6. Detector fine tuning
    # --------------------------------------------------
    unfreeze_last_backbone_block(model)
    print("\nTrainable parameters after unfreezing last backbone block:", count_trainable_params(model))

    finetune_optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=FINETUNE_LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )

    finetune_losses = []
    for epoch in range(FINETUNE_EPOCHS):
        epoch_loss = train_one_epoch(model, train_loader, finetune_optimizer, DEVICE)
        finetune_losses.append(epoch_loss)
        print(f"Fine-tune Epoch [{epoch+1}/{FINETUNE_EPOCHS}] Loss: {epoch_loss:.4f}")

    plot_losses(baseline_losses, finetune_losses)

    finetuned_model = copy.deepcopy(model)

    print("\nFine-tuned predictions on same 5 test images:")
    visualize_predictions_on_indices(finetuned_model, test_dataset, test_indices_global, DEVICE, conf_thresh=CONF_THRESH_DEFAULT, title_prefix="Fine-tuned")

    # --------------------------------------------------
    # TASK 7. IoU based matching of predictions
    # --------------------------------------------------
    one_image, one_target = test_dataset[0]
    one_output = predict(finetuned_model, [one_image], DEVICE)[0]
    one_output = filter_predictions(one_output, conf_thresh=CONF_THRESH_DEFAULT)

    pred_boxes = one_output["boxes"]
    gt_boxes = one_target["boxes"]

    iou_matrix, matched_pairs, tp, fp, fn = compute_matches_for_image(pred_boxes, gt_boxes, iou_thresh=IOU_MATCH_THRESHOLD)
    print("\nTask 7. IoU matrix:")
    print(iou_matrix)
    print("Matched pairs (pred_idx, gt_idx, iou):")
    for item in matched_pairs:
        print(item)
    print(f"TP={tp}, FP={fp}, FN={fn}")

    # --------------------------------------------------
    # TASK 8. Precision recall AP evaluation
    # --------------------------------------------------
    selected_class_id = CLASS_TO_IDX["with_mask"]
    recalls, precisions, ap = compute_ap_for_class(
        finetuned_model,
        test_loader,
        selected_class_id,
        DEVICE,
        iou_thresh=0.5
    )

    print(f"\nTask 8. AP for class '{CLASS_NAMES[selected_class_id]}' at IoU=0.5: {ap:.4f}")
    plt.figure(figsize=(6, 5))
    plt.plot(recalls, precisions, marker='o')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f"Precision-Recall Curve: {CLASS_NAMES[selected_class_id]}")
    plt.grid(True)
    plt.show()

    # --------------------------------------------------
    # TASK 9. Mean Average Precision evaluation
    # --------------------------------------------------
    baseline_ap_per_class, baseline_map = compute_map(baseline_model, test_loader, DEVICE, iou_thresh=0.5)
    finetuned_ap_per_class, finetuned_map = compute_map(finetuned_model, test_loader, DEVICE, iou_thresh=0.5)

    print("\nTask 9. Baseline AP per class:")
    for k, v in baseline_ap_per_class.items():
        print(f"{k}: {v:.4f}")
    print(f"Baseline mAP@0.5: {baseline_map:.4f}")

    print("\nFine-tuned AP per class:")
    for k, v in finetuned_ap_per_class.items():
        print(f"{k}: {v:.4f}")
    print(f"Fine-tuned mAP@0.5: {finetuned_map:.4f}")

    print("\nComparison table:")
    print("Class\t\tBaseline AP\tFine-tuned AP")
    for cls_name in CLASS_NAMES[1:]:
        print(f"{cls_name}\t{baseline_ap_per_class[cls_name]:.4f}\t\t{finetuned_ap_per_class[cls_name]:.4f}")
    print(f"mAP@0.5\t{baseline_map:.4f}\t\t{finetuned_map:.4f}")

    # --------------------------------------------------
    # TASK 10. Confidence threshold analysis
    # --------------------------------------------------
    thresholds = [0.3, 0.5, 0.7]
    selected_image, _ = test_dataset[0]
    selected_output = predict(finetuned_model, [selected_image], DEVICE)[0]

    for thr in thresholds:
        filtered = filter_predictions(selected_output, conf_thresh=thr)
        print(f"Confidence threshold {thr}: {len(filtered['boxes'])} detections")
        show_image_with_boxes(
            selected_image,
            filtered["boxes"].tolist(),
            filtered["labels"].tolist(),
            filtered["scores"].tolist(),
            title=f"Threshold = {thr}"
        )

    # --------------------------------------------------
    # TASK 11. Non maximum suppression analysis
    # --------------------------------------------------
    raw_output = predict(finetuned_model, [selected_image], DEVICE)[0]
    nms_a = run_custom_nms_analysis(raw_output, conf_thresh=0.05, nms_thresh=0.3)
    nms_b = run_custom_nms_analysis(raw_output, conf_thresh=0.05, nms_thresh=0.7)

    print("\nTask 11. NMS analysis")
    print(f"NMS threshold 0.3 -> before: {nms_a['before_nms']}, after: {nms_a['after_nms']}")
    print(f"NMS threshold 0.7 -> before: {nms_b['before_nms']}, after: {nms_b['after_nms']}")

    show_image_with_boxes(
        selected_image,
        nms_a["boxes"].tolist(),
        nms_a["labels"].tolist(),
        nms_a["scores"].tolist(),
        title="After NMS threshold = 0.3"
    )
    show_image_with_boxes(
        selected_image,
        nms_b["boxes"].tolist(),
        nms_b["labels"].tolist(),
        nms_b["scores"].tolist(),
        title="After NMS threshold = 0.7"
    )

    # --------------------------------------------------
    # TASK 12. Inference speed evaluation
    # --------------------------------------------------
    baseline_speed = evaluate_inference_speed(baseline_model, test_loader, DEVICE)
    finetuned_speed = evaluate_inference_speed(finetuned_model, test_loader, DEVICE)

    print("\nTask 12. Inference speed evaluation")
    print("Hardware:", DEVICE)
    print(f"Average inference time per image (baseline): {baseline_speed:.4f} sec")
    print(f"Average inference time per image (fine-tuned): {finetuned_speed:.4f} sec")

    # --------------------------------------------------
    # TASK 13. Error analysis with conclusions
    # --------------------------------------------------
    print("\nTask 13. Error analysis examples")
    # Show up to 6 examples: 3 likely correct + 3 likely failures
    for idx in range(min(6, len(test_dataset))):
        image, target = test_dataset[idx]
        output = predict(finetuned_model, [image], DEVICE)[0]
        output = filter_predictions(output, conf_thresh=0.5)
        visualize_gt_and_pred(
            image=image,
            gt_boxes=target["boxes"],
            gt_labels=target["labels"],
            pred_boxes=output["boxes"],
            pred_labels=output["labels"],
            pred_scores=output["scores"],
            title=f"Error analysis sample {idx}"
        )

    print("Suggested conclusions for report:")
    print("1. Frozen backbone training learns detector head faster but is limited in adaptation.")
    print("2. Fine-tuning the last backbone block usually improves localization and classification quality.")
    print("3. Lower confidence thresholds increase detections but can increase false positives.")
    print("4. NMS removes duplicate overlapping boxes and keeps final detections cleaner.")

    # --------------------------------------------------
    # TASK 14. Anchor and box prior analysis
    # --------------------------------------------------
    print("\nTask 14. Anchor analysis")
    inspect_anchor_generator(finetuned_model)

    # --------------------------------------------------
    # TASK 15. Backbone feature level comparison
    # --------------------------------------------------
    print("\nTask 15. Backbone feature level comparison")
    feature_dict_2 = extract_backbone_features(finetuned_model, selected_image, DEVICE)
    compare_three_backbone_levels(feature_dict_2)


if __name__ == "__main__":
    main()
