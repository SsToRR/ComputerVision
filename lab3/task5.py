import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# Fixed seed
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing (same as previous tasks)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# Load dataset
full_dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

class_names = full_dataset.classes

# Multiclass task (0,1,2)
multi_classes = [0, 1, 2]
multi_indices = [i for i, (_, label) in enumerate(full_dataset)
                 if label in multi_classes]
multi_dataset = Subset(full_dataset, multi_indices)

train_size = int(0.7 * len(multi_dataset))
test_size = len(multi_dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(
    multi_dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(seed)
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Baseline CNN (same as Task 3)
class BaselineCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(BaselineCNN, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(32 * 8 * 8, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = BaselineCNN(num_classes=3).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Quick training (5 epochs)
for epoch in range(5):
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


# Evaluation
model.eval()
all_preds = []
all_labels = []
all_images = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_images.extend(images.cpu())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:\n", cm)

# Plot confusion matrix
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks(range(3), multi_classes)
plt.yticks(range(3), multi_classes)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Show correctly classified examples
correct_indices = np.where(all_preds == all_labels)[0]
misclassified_indices = np.where(all_preds != all_labels)[0]

plt.figure(figsize=(10,4))
for i in range(5):
    idx = correct_indices[i]
    plt.subplot(2,5,i+1)
    plt.imshow(all_images[idx].squeeze(), cmap='gray')
    plt.title(f"T:{all_labels[idx]} P:{all_preds[idx]}")
    plt.axis("off")

for i in range(5):
    idx = misclassified_indices[i]
    plt.subplot(2,5,i+6)
    plt.imshow(all_images[idx].squeeze(), cmap='gray')
    plt.title(f"T:{all_labels[idx]} P:{all_preds[idx]}")
    plt.axis("off")

plt.tight_layout()
plt.show()
