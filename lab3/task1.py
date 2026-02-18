import torch
import numpy as np
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# Fixed random seed
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Transform
transform = transforms.ToTensor()

# Load Fashion MNIST training dataset
full_dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

total_images = len(full_dataset)

# Compute class distribution
class_counts = {i: 0 for i in range(10)}
for _, label in full_dataset:
    class_counts[label] += 1

# -------- Binary task (class 0 and 1) --------
binary_classes = [0, 1]
binary_indices = [i for i, (_, label) in enumerate(full_dataset)
                  if label in binary_classes]
binary_dataset = Subset(full_dataset, binary_indices)

# -------- Multiclass task (class 0,1,2) --------
multi_classes = [0, 1, 2]
multi_indices = [i for i, (_, label) in enumerate(full_dataset)
                 if label in multi_classes]
multi_dataset = Subset(full_dataset, multi_indices)

# 70% split
def split_dataset(dataset, train_ratio=0.7):
    size = len(dataset)
    train_size = int(train_ratio * size)
    test_size = size - train_size
    return torch.utils.data.random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

binary_train, binary_test = split_dataset(binary_dataset)
multi_train, multi_test = split_dataset(multi_dataset)

batch_size = 32

binary_loader = DataLoader(binary_train, batch_size=batch_size, shuffle=True)
multi_loader = DataLoader(multi_train, batch_size=batch_size, shuffle=True)

binary_images, _ = next(iter(binary_loader))
multi_images, _ = next(iter(multi_loader))

print("Dataset name: Fashion MNIST")
print("Source: Zalando Research (available in torchvision)")
print("Total images:", total_images)
print("Class distribution:", class_counts)
print("Binary task number of classes:", len(binary_classes))
print("Multiclass task number of classes:", len(multi_classes))
print("Single image shape:", full_dataset[0][0].shape)
print("Binary training batch shape:", binary_images.shape)
print("Multiclass training batch shape:", multi_images.shape)
