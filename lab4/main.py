import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt

# -------------------------
# Global settings
# -------------------------
SEED = 42
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 128
NUM_WORKERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS_FROZEN = 5
EPOCHS_FINETUNE = 3

MODEL_FROZEN_PATH = "frozen.pth"
MODEL_FINETUNE_PATH = "finetune.pth"
RESULTS_FILE = "results.txt"

# -------------------------
# Seed
# -------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -------------------------
# Dataset
# -------------------------
def prepare_data():
    weights = ResNet18_Weights.DEFAULT

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=weights.transforms().mean,
                             std=weights.transforms().std)
    ])

    train_dataset = ImageFolder("Vegetable Images/train", transform=transform)
    test_dataset = ImageFolder("Vegetable Images/test", transform=transform)

    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=NUM_WORKERS)

    test_loader = DataLoader(test_dataset,
                         batch_size=EVAL_BATCH_SIZE,
                         shuffle=False,
                         num_workers=NUM_WORKERS)

    return train_loader, test_loader, train_dataset.classes

# -------------------------
# Training utilities
# -------------------------
def train_model(model, train_loader, test_loader, epochs, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    history = {"train_acc": [], "test_acc": []}

    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(out, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_acc = correct / total

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                preds = torch.argmax(out, 1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        test_acc = correct / total

        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        print(f"Epoch {epoch+1}/{epochs}  Train Acc={train_acc:.4f}  Test Acc={test_acc:.4f}")

    return history

# -------------------------
# Manual metrics
# -------------------------
def get_predictions(model, loader):
    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            out = model(x)
            preds = torch.argmax(out, 1).cpu().numpy()

            y_true.extend(y.numpy())
            y_pred.extend(preds)

    return np.array(y_true), np.array(y_pred)

def confusion_matrix_manual(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def metrics_from_cm(cm):
    accuracy = np.trace(cm) / np.sum(cm)

    precisions = []
    recalls = []
    f1s = []

    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return accuracy, np.mean(precisions), np.mean(recalls), np.mean(f1s)

# -------------------------
# Main
# -------------------------
def main():
    set_seed(SEED)

    train_loader, test_loader, class_names = prepare_data()
    num_classes = len(class_names)

    weights = ResNet18_Weights.DEFAULT

    model_frozen = resnet18(weights=weights)
    model_frozen.fc = nn.Linear(model_frozen.fc.in_features, num_classes)
    model_frozen.to(DEVICE)

    model_finetune = resnet18(weights=weights)
    model_finetune.fc = nn.Linear(model_finetune.fc.in_features, num_classes)
    model_finetune.to(DEVICE)

    if not os.path.exists(MODEL_FROZEN_PATH):

        for param in model_frozen.parameters():
            param.requires_grad = False
        for param in model_frozen.fc.parameters():
            param.requires_grad = True

        train_model(model_frozen, train_loader, test_loader, EPOCHS_FROZEN, 1e-3)
        torch.save(model_frozen.state_dict(), MODEL_FROZEN_PATH)

    else:
        model_frozen.load_state_dict(torch.load(MODEL_FROZEN_PATH))

    if not os.path.exists(MODEL_FINETUNE_PATH):

        for param in model_finetune.parameters():
            param.requires_grad = False
        for param in model_finetune.layer4.parameters():
            param.requires_grad = True
        for param in model_finetune.fc.parameters():
            param.requires_grad = True

        train_model(model_finetune, train_loader, test_loader, EPOCHS_FINETUNE, 1e-4)
        torch.save(model_finetune.state_dict(), MODEL_FINETUNE_PATH)

    else:
        model_finetune.load_state_dict(torch.load(MODEL_FINETUNE_PATH))

    y_true_frozen, y_pred_frozen = get_predictions(model_frozen, test_loader)
    y_true_finetune, y_pred_finetune = get_predictions(model_finetune, test_loader)

    cm_frozen = confusion_matrix_manual(y_true_frozen, y_pred_frozen, num_classes)
    cm_finetune = confusion_matrix_manual(y_true_finetune, y_pred_finetune, num_classes)

    metrics_frozen = metrics_from_cm(cm_frozen)
    metrics_finetune = metrics_from_cm(cm_finetune)

    with open(RESULTS_FILE, "w") as f:
        f.write("Task 5 Results\n")
        f.write(f"Frozen: {metrics_frozen}\n")
        f.write(f"Finetune: {metrics_finetune}\n")

    print("Results saved to", RESULTS_FILE)

if __name__ == "__main__":
    main()
