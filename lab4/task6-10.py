import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt

# =========================
# CONFIG (edit only here)
# =========================
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_TEST_DIR = "Vegetable Images/test"
DATA_TRAIN_DIR = "Vegetable Images/train"   # used only for Task 7 scratch training

FROZEN_PATH = "frozen.pth"
FINETUNE_PATH = "finetune.pth"
SCRATCH_PATH = "scratch.pth"

OUT_DIR = "lab_outputs_tasks6_10"
os.makedirs(OUT_DIR, exist_ok=True)

# Speed controls
EVAL_SUBSET_SIZE = None          # Set None for full test set.
EVAL_BATCH_SIZE = 128            # large batch is faster in eval
NUM_WORKERS = 2                  # if Windows issues, set 0

# Task 7 scratch training speed controls
TRAIN_SUBSET_SIZE = None         # set e.g. 6000 to train scratch on fewer images; None uses full train set
TRAIN_BATCH_SIZE = 64
EPOCHS_SCRATCH = 6               # increase if teacher requires more training
LR_SCRATCH = 1e-3

# =========================
# Reproducibility
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# =========================
# Dataset + transforms
# =========================
def build_transforms():
    weights = ResNet18_Weights.DEFAULT
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
    ])
    return tfm, weights

def load_datasets():
    tfm, _ = build_transforms()

    test_ds = ImageFolder(DATA_TEST_DIR, transform=tfm)
    class_names = test_ds.classes

    test_ds_eval = test_ds

    test_loader = DataLoader(
        test_ds_eval,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    return test_ds, test_ds_eval, test_loader, class_names

# =========================
# Models
# =========================
def build_resnet_for_dataset(num_classes: int):
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def load_saved_resnet(path: str, num_classes: int):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing file: {path}. You need to run Tasks 3–4 training once to create it."
        )
    model = build_resnet_for_dataset(num_classes).to(DEVICE)
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

# =========================
# Manual metrics
# =========================
@torch.no_grad()
def get_predictions(model: nn.Module, loader: DataLoader):
    model.eval()
    all_true = []
    all_pred = []

    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        logits = model(x)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        all_true.append(y.numpy())
        all_pred.append(preds)

    return np.concatenate(all_true), np.concatenate(all_pred)

def confusion_matrix_manual(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm

def metrics_from_confusion_matrix(cm: np.ndarray):
    total = cm.sum()
    correct = np.trace(cm)
    accuracy = correct / total if total > 0 else 0.0

    precisions, recalls, f1s = [], [], []
    k = cm.shape[0]

    for c in range(k):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return float(accuracy), float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1s))

# =========================
# Task 6: Confusion matrix + examples
# =========================
def save_confusion_matrix_plot(cm: np.ndarray, class_names: list, title: str, out_path: str):
    plt.figure(figsize=(9, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def save_six_images(model: nn.Module, dataset_eval: Subset, class_names: list, out_path: str):
    model.eval()
    correct_items = []
    wrong_items = []

    tfm, weights = build_transforms()
    inv_norm = transforms.Normalize(
        mean=[-m/s for m, s in zip(weights.transforms().mean, weights.transforms().std)],
        std=[1/s for s in weights.transforms().std],
    )

    for idx in range(len(dataset_eval)):
        x, y = dataset_eval[idx]
        x_in = x.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = int(torch.argmax(model(x_in), dim=1).item())

        if pred == y and len(correct_items) < 3:
            correct_items.append((x, int(y), pred))
        if pred != y and len(wrong_items) < 3:
            wrong_items.append((x, int(y), pred))

        if len(correct_items) == 3 and len(wrong_items) == 3:
            break

    items = correct_items + wrong_items
    plt.figure(figsize=(10, 6))

    for i, (img, y, pred) in enumerate(items, start=1):
        plt.subplot(2, 3, i)
        img_vis = inv_norm(img).clamp(0, 1)
        plt.imshow(np.transpose(img_vis.numpy(), (1, 2, 0)))
        plt.title(f"true={class_names[y]}\npred={class_names[pred]}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# =========================
# Task 7: Scratch model (small CNN) + saving
# =========================
class SmallCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)

def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_scratch_if_needed(num_classes: int, class_names: list):
    if os.path.exists(SCRATCH_PATH):
        model = SmallCNN(num_classes).to(DEVICE)
        model.load_state_dict(torch.load(SCRATCH_PATH, map_location=DEVICE))
        model.eval()
        return model, None

    tfm, _ = build_transforms()
    train_ds = ImageFolder(DATA_TRAIN_DIR, transform=tfm)

    if TRAIN_SUBSET_SIZE is not None:
        n = min(TRAIN_SUBSET_SIZE, len(train_ds))
        train_ds = Subset(train_ds, list(range(n)))

    train_loader = DataLoader(
        train_ds,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    model = SmallCNN(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_SCRATCH)

    history = {"train_loss": [], "epoch_time": []}

    for epoch in range(1, EPOCHS_SCRATCH + 1):
        t0 = time.perf_counter()
        model.train()
        running_loss = 0.0
        count = 0

        for x, y in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            count += y.size(0)

        t1 = time.perf_counter()
        epoch_loss = running_loss / max(count, 1)
        history["train_loss"].append(float(epoch_loss))
        history["epoch_time"].append(float(t1 - t0))
        print(f"Scratch Epoch {epoch:02d}/{EPOCHS_SCRATCH} | train loss={epoch_loss:.4f} | time={t1-t0:.2f}s")

    torch.save(model.state_dict(), SCRATCH_PATH)
    model.eval()
    return model, history

# =========================
# Task 8: Feature maps
# =========================
def extract_feature_maps_resnet18(model: nn.Module, x: torch.Tensor):
    activations = {}

    def hook(name):
        def _fn(module, inp, out):
            activations[name] = out.detach().cpu()
        return _fn

    h1 = model.conv1.register_forward_hook(hook("early"))
    h2 = model.layer4[1].conv2.register_forward_hook(hook("deep"))

    model.eval()
    with torch.no_grad():
        _ = model(x.to(DEVICE))

    h1.remove()
    h2.remove()
    return activations["early"], activations["deep"]

def save_four_feature_maps(feature_tensor: torch.Tensor, title: str, out_path: str):
    fmap = feature_tensor[0]  # [C,H,W]
    plt.figure(figsize=(10, 3))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.imshow(fmap[i].numpy(), interpolation="nearest")
        plt.title(f"{title} #{i}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# =========================
# Task 9: Efficiency analysis
# =========================
def measure_forward_time(model: nn.Module, loader: DataLoader, warmup_batches: int = 2, measure_batches: int = 10):
    model.eval()
    times = []
    it = iter(loader)

    with torch.no_grad():
        for _ in range(warmup_batches):
            try:
                x, _ = next(it)
            except StopIteration:
                return None
            _ = model(x.to(DEVICE, non_blocking=True))

        for _ in range(measure_batches):
            try:
                x, _ = next(it)
            except StopIteration:
                break
            t0 = time.perf_counter()
            _ = model(x.to(DEVICE, non_blocking=True))
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

    return float(np.mean(times)) if times else None

# =========================
# Task 10: Summary saving
# =========================
def write_results_json(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

# =========================
# MAIN: Tasks 6–10
# =========================
def main():
    set_seed(SEED)

    test_ds_full, test_ds_eval, test_loader, class_names = load_datasets()
    num_classes = len(class_names)

    model_frozen = load_saved_resnet(FROZEN_PATH, num_classes)
    model_finetune = load_saved_resnet(FINETUNE_PATH, num_classes)

    # -------- Task 6 depends on picking better model from Task 5 metrics ----------
    # We compute manual metrics quickly here (subset), then proceed to Task 6.
    t0 = time.perf_counter()
    y_true_frozen, y_pred_frozen = get_predictions(model_frozen, test_loader)
    y_true_finetune, y_pred_finetune = get_predictions(model_finetune, test_loader)
    t1 = time.perf_counter()

    cm_frozen = confusion_matrix_manual(y_true_frozen, y_pred_frozen, num_classes)
    cm_finetune = confusion_matrix_manual(y_true_finetune, y_pred_finetune, num_classes)

    acc_frozen, prec_frozen, rec_frozen, f1_frozen = metrics_from_confusion_matrix(cm_frozen)
    acc_finetune, prec_finetune, rec_finetune, f1_finetune = metrics_from_confusion_matrix(cm_finetune)

    better_model = model_finetune if acc_finetune >= acc_frozen else model_frozen
    better_cm = cm_finetune if acc_finetune >= acc_frozen else cm_frozen
    better_name = "finetune" if acc_finetune >= acc_frozen else "frozen"

    # -------- Task 6: Confusion matrix + example images ----------
    cm_path = os.path.join(OUT_DIR, f"task6_confusion_matrix_{better_name}.png")
    save_confusion_matrix_plot(better_cm, class_names, f"Task 6 Confusion Matrix ({better_name})", cm_path)

    six_path = os.path.join(OUT_DIR, f"task6_examples_{better_name}.png")
    save_six_images(better_model, test_ds_eval, class_names, six_path)

    # -------- Task 7: Train scratch model (cached) + metrics ----------
    scratch_model, scratch_history = train_scratch_if_needed(num_classes, class_names)

    y_true_scratch, y_pred_scratch = get_predictions(scratch_model, test_loader)
    cm_scratch = confusion_matrix_manual(y_true_scratch, y_pred_scratch, num_classes)
    acc_scratch, prec_scratch, rec_scratch, f1_scratch = metrics_from_confusion_matrix(cm_scratch)

    # -------- Task 8: Feature maps for one test image ----------
    one_img, one_label = test_ds_eval[0]
    x_one = one_img.unsqueeze(0)

    early_act, deep_act = extract_feature_maps_resnet18(better_model, x_one)
    early_path = os.path.join(OUT_DIR, f"task8_featuremaps_early_{better_name}.png")
    deep_path = os.path.join(OUT_DIR, f"task8_featuremaps_deep_{better_name}.png")
    save_four_feature_maps(early_act, "Early layer (conv1)", early_path)
    save_four_feature_maps(deep_act, "Deep layer (layer4.1.conv2)", deep_path)

    # -------- Task 9: Efficiency analysis ----------
    pretrained_params_trainable = count_trainable_params(better_model)
    scratch_params_trainable = count_trainable_params(scratch_model)

    ft_time = measure_forward_time(better_model, test_loader)
    sc_time = measure_forward_time(scratch_model, test_loader)

    # -------- Task 10: Final summary saved ----------
    results = {
        "settings": {
            "device": str(DEVICE),
            "eval_subset_size": EVAL_SUBSET_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "num_classes": num_classes
        },
        "task5_like_metrics_on_eval_subset": {
            "frozen": {"accuracy": acc_frozen, "macro_precision": prec_frozen, "macro_recall": rec_frozen, "macro_f1": f1_frozen},
            "finetune": {"accuracy": acc_finetune, "macro_precision": prec_finetune, "macro_recall": rec_finetune, "macro_f1": f1_finetune},
            "better_model": better_name,
            "prediction_time_seconds": float(t1 - t0)
        },
        "task6_outputs": {
            "confusion_matrix_image": cm_path,
            "six_examples_image": six_path
        },
        "task7_scratch": {
            "scratch_model_saved": SCRATCH_PATH,
            "accuracy": acc_scratch,
            "macro_precision": prec_scratch,
            "macro_recall": rec_scratch,
            "macro_f1": f1_scratch,
            "history": scratch_history
        },
        "task8_outputs": {
            "early_feature_maps_image": early_path,
            "deep_feature_maps_image": deep_path
        },
        "task9_efficiency": {
            "better_model_name": better_name,
            "better_model_trainable_params": pretrained_params_trainable,
            "scratch_trainable_params": scratch_params_trainable,
            "mean_forward_time_seconds_better_model": ft_time,
            "mean_forward_time_seconds_scratch": sc_time
        }
    }

    out_json = os.path.join(OUT_DIR, "tasks6_10_results.json")
    write_results_json(out_json, results)

    out_txt = os.path.join(OUT_DIR, "tasks6_10_summary.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("Tasks 6–10 Summary\n\n")
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Eval subset size: {EVAL_SUBSET_SIZE}\n\n")
        f.write("Frozen metrics (subset):\n")
        f.write(f"Accuracy={acc_frozen:.6f}, Precision={prec_frozen:.6f}, Recall={rec_frozen:.6f}, MacroF1={f1_frozen:.6f}\n\n")
        f.write("Finetune metrics (subset):\n")
        f.write(f"Accuracy={acc_finetune:.6f}, Precision={prec_finetune:.6f}, Recall={rec_finetune:.6f}, MacroF1={f1_finetune:.6f}\n\n")
        f.write(f"Selected better model for Task 6/8: {better_name}\n\n")
        f.write("Scratch model metrics (subset):\n")
        f.write(f"Accuracy={acc_scratch:.6f}, Precision={prec_scratch:.6f}, Recall={rec_scratch:.6f}, MacroF1={f1_scratch:.6f}\n\n")
        f.write("Efficiency:\n")
        f.write(f"Better model trainable params: {pretrained_params_trainable}\n")
        f.write(f"Scratch trainable params: {scratch_params_trainable}\n")
        f.write(f"Mean forward time better model (s): {ft_time}\n")
        f.write(f"Mean forward time scratch (s): {sc_time}\n\n")
        f.write("Saved figures:\n")
        f.write(f"{cm_path}\n{six_path}\n{early_path}\n{deep_path}\n")

    print("Tasks 6–10 finished.")
    print("Saved outputs in:", OUT_DIR)
    print("Key files:")
    print(" -", out_json)
    print(" -", out_txt)
    print(" -", cm_path)
    print(" -", six_path)
    print(" -", early_path)
    print(" -", deep_path)

if __name__ == "__main__":
    main()
