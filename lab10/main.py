# =========================
# Imports
# =========================
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# === Task 1: Dataset ===
# =========================

transform = transforms.Compose([
    transforms.ToTensor()  # automatically scales to [0,1]
])

train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

print("Train size:", len(train_dataset))
print("Test size:", len(test_dataset))

# Show images
def show_images(images, title):
    images = images[:8]
    fig, axes = plt.subplots(1, 8, figsize=(12,2))
    for i in range(8):
        axes[i].imshow(images[i].squeeze(), cmap='gray')
        axes[i].axis('off')
    plt.suptitle(title)
    plt.show()

data_iter = iter(train_loader)
images, _ = next(data_iter)
show_images(images, "Original Images")

# =========================
# === Task 2: Autoencoder ===
# =========================

latent_dim = 16  # important parameter

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        out = self.decoder(latent)
        out = out.view(-1, 1, 28, 28)
        return out

model = Autoencoder().to(device)

print(model)

# Count parameters
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Trainable parameters:", params)
print("Latent dimension:", latent_dim)

# =========================
# === Task 3: Before Training ===
# =========================

criterion = nn.MSELoss()

model.eval()
with torch.no_grad():
    images, _ = next(iter(test_loader))
    images = images.to(device)

    outputs = model(images)
    loss = criterion(outputs, images)

    print("Loss before training:", loss.item())

    show_images(images.cpu(), "Original Before Training")
    show_images(outputs.cpu(), "Reconstructed Before Training")

# =========================
# === Task 4: Training ===
# =========================

optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5
train_losses = []
test_losses = []

for epoch in range(epochs):
    model.train()
    train_loss = 0

    for images, _ in train_loader:
        images = images.to(device)

        outputs = model(images)
        loss = criterion(outputs, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # Test loss
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, images)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    test_losses.append(test_loss)

    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}")

# Plot losses
plt.plot(train_losses, label='Train')
plt.plot(test_losses, label='Test')
plt.legend()
plt.title("Loss Curve")
plt.show()

# =========================
# === Task 5: After Training ===
# =========================

model.eval()
with torch.no_grad():
    images, _ = next(iter(test_loader))
    images = images.to(device)

    outputs = model(images)

    mse_list = ((outputs - images) ** 2).mean(dim=[1,2,3])
    print("Average MSE:", mse_list.mean().item())

    show_images(images.cpu(), "Original After Training")
    show_images(outputs.cpu(), "Reconstructed After Training")

# =========================
# === Task 6: Latent Space ===
# =========================

model.eval()
latents = []
labels = []

with torch.no_grad():
    for images, lbls in test_loader:
        images = images.to(device)
        z = model.encoder(images)
        latents.append(z.cpu().numpy())
        labels.append(lbls.numpy())

latents = np.concatenate(latents)
labels = np.concatenate(labels)

print("Latent shape:", latents.shape)

# PCA visualization
pca = PCA(n_components=2)
latents_pca = pca.fit_transform(latents)

plt.scatter(latents_pca[:,0], latents_pca[:,1], c=labels, cmap='tab10')
plt.colorbar()
plt.title("Latent Space PCA")
plt.show()

# =========================
# === Task 7: Latent Dim Effect ===
# =========================

print("Try latent_dim = 8 and latent_dim = 32 manually and compare losses")

# =========================
# === Task 8: Denoising AE ===
# =========================

def add_noise(x):
    noise = torch.randn_like(x) * 0.3
    return torch.clamp(x + noise, 0, 1)

# Train briefly for denoising
for epoch in range(2):
    for images, _ in train_loader:
        images = images.to(device)
        noisy = add_noise(images)

        outputs = model(noisy)
        loss = criterion(outputs, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Show results
model.eval()
with torch.no_grad():
    images, _ = next(iter(test_loader))
    images = images.to(device)
    noisy = add_noise(images)
    outputs = model(noisy)

    show_images(noisy.cpu(), "Noisy Images")
    show_images(outputs.cpu(), "Denoised Images")

# =========================
# === Task 9: Anomaly Detection ===
# =========================

# Train only on digit "1"
filtered = [(img, lbl) for img, lbl in train_dataset if lbl == 1]
filtered_loader = torch.utils.data.DataLoader(filtered, batch_size=128, shuffle=True)

# quick retrain
for epoch in range(3):
    for images, _ in filtered_loader:
        images = images.to(device)

        outputs = model(images)
        loss = criterion(outputs, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Compute reconstruction error
errors = []
labels = []

with torch.no_grad():
    for images, lbls in test_loader:
        images = images.to(device)
        outputs = model(images)

        err = ((outputs - images) ** 2).mean(dim=[1,2,3])
        errors.extend(err.cpu().numpy())
        labels.extend(lbls.numpy())

errors = np.array(errors)
labels = np.array(labels)

# Plot
plt.hist(errors[labels == 1], bins=50, alpha=0.5, label='Normal (1)')
plt.hist(errors[labels != 1], bins=50, alpha=0.5, label='Anomaly')
plt.legend()
plt.title("Reconstruction Error Distribution")
plt.show()