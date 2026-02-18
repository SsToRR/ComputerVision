import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Fixed random seed
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Compute mean and std manually
raw_dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

all_pixels = torch.cat([img.view(-1) for img, _ in raw_dataset])
mean = all_pixels.mean()
std = all_pixels.std()

print("Computed mean:", mean.item())
print("Computed std:", std.item())

# Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,))
])

dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=False,
    transform=preprocess
)

loader = DataLoader(dataset, batch_size=32, shuffle=True)

images, _ = next(iter(loader))

print("Training batch shape after preprocessing:", images.shape)

# Show example before and after preprocessing
original_transform = transforms.ToTensor()

original_dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=False,
    transform=original_transform
)

original_image, _ = original_dataset[0]
processed_image, _ = dataset[0]

plt.figure(figsize=(6,3))

plt.subplot(1,2,1)
plt.imshow(original_image.squeeze(), cmap='gray')
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(processed_image.squeeze(), cmap='gray')
plt.title("After preprocessing")
plt.axis("off")

plt.show()
