# =========================================================
# GAN LAB IMPLEMENTATION (Computer Vision)
# =========================================================

# =========================
# IMPORTS
# =========================
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# HYPERPARAMETERS
# =========================
BATCH_SIZE = 128
Z_DIM = 100
LR = 0.0002
EPOCHS = 20
IMAGE_SIZE = 32

# =========================================================
# ==== TASK 1: DATASET ====================================
# =========================================================

# Transform images to [-1, 1]
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Load CIFAR-10 dataset (only cars class = label 1)
dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    transform=transform,
    download=True
)

# Filter only "car" images (label = 1)
indices = [i for i, label in enumerate(dataset.targets) if label == 1]
dataset = torch.utils.data.Subset(dataset, indices)

loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Dataset size:", len(dataset))

# =========================================================
# ==== TASK 2: GENERATOR ==================================
# =========================================================

class Generator(nn.Module):
    """
    Generator: converts noise vector into image
    """
    def __init__(self, z_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 3 * IMAGE_SIZE * IMAGE_SIZE),
            nn.Tanh()
        )

    def forward(self, x):
        img = self.net(x)
        return img.view(-1, 3, IMAGE_SIZE, IMAGE_SIZE)


gen = Generator(Z_DIM).to(device)

# Print architecture and params
print(gen)
print("Generator params:", sum(p.numel() for p in gen.parameters()))

# Generate fake images BEFORE training
noise = torch.randn(16, Z_DIM).to(device)
fake_images = gen(noise).detach().cpu()

# Show fake images
grid = torchvision.utils.make_grid(fake_images, normalize=True)
plt.imshow(np.transpose(grid, (1, 2, 0)))
plt.title("Fake images BEFORE training")
plt.show()

# =========================================================
# ==== TASK 3: DISCRIMINATOR ===============================
# =========================================================

class Discriminator(nn.Module):
    """
    Discriminator: classifies image as real or fake
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * IMAGE_SIZE * IMAGE_SIZE, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


disc = Discriminator().to(device)

print(disc)
print("Discriminator params:", sum(p.numel() for p in disc.parameters()))

# Test discriminator
real_batch, _ = next(iter(loader))
real_batch = real_batch.to(device)

fake_batch = gen(torch.randn(BATCH_SIZE, Z_DIM).to(device))

real_score = disc(real_batch)
fake_score = disc(fake_batch)

print("Real scores:", real_score[:5])
print("Fake scores:", fake_score[:5])

# =========================================================
# ==== TASK 4: TRAINING LOOP ===============================
# =========================================================

criterion = nn.BCELoss()
opt_gen = optim.Adam(gen.parameters(), lr=LR)
opt_disc = optim.Adam(disc.parameters(), lr=LR)

g_losses = []
d_losses = []

for epoch in range(EPOCHS):
    for real, _ in loader:
        real = real.to(device)
        batch_size = real.shape[0]

        # === Train Discriminator ===
        noise = torch.randn(batch_size, Z_DIM).to(device)
        fake = gen(noise)

        disc_real = disc(real).view(-1)
        loss_real = criterion(disc_real, torch.ones_like(disc_real))

        disc_fake = disc(fake.detach()).view(-1)
        loss_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        loss_disc = (loss_real + loss_fake) / 2

        opt_disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # === Train Generator ===
        output = disc(fake).view(-1)
        loss_gen = criterion(output, torch.ones_like(output))

        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

    g_losses.append(loss_gen.item())
    d_losses.append(loss_disc.item())

    print(f"Epoch {epoch+1}/{EPOCHS} | G Loss: {loss_gen:.4f} | D Loss: {loss_disc:.4f}")

# Plot losses
plt.plot(g_losses, label="Generator")
plt.plot(d_losses, label="Discriminator")
plt.legend()
plt.title("Loss curves")
plt.show()

# =========================================================
# ==== TASK 5: GENERATE IMAGES =============================
# =========================================================

def show_images(generator, title):
    noise = torch.randn(16, Z_DIM).to(device)
    fake = generator(noise).detach().cpu()
    grid = torchvision.utils.make_grid(fake, normalize=True)
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.title(title)
    plt.show()

show_images(gen, "Generated images AFTER training")

# =========================================================
# ==== TASK 6: LATENT INTERPOLATION ========================
# =========================================================

z1 = torch.randn(1, Z_DIM).to(device)
z2 = torch.randn(1, Z_DIM).to(device)

interpolated = []
for alpha in np.linspace(0, 1, 10):
    z = (1 - alpha) * z1 + alpha * z2
    interpolated.append(gen(z).detach().cpu())

interpolated = torch.cat(interpolated)
grid = torchvision.utils.make_grid(interpolated, nrow=10, normalize=True)

plt.imshow(np.transpose(grid, (1, 2, 0)))
plt.title("Latent interpolation")
plt.show()

# =========================================================
# ==== TASK 7: MODE COLLAPSE ===============================
# =========================================================

# Generate many images
noise = torch.randn(64, Z_DIM).to(device)
samples = gen(noise).detach().cpu()

grid = torchvision.utils.make_grid(samples, normalize=True)
plt.imshow(np.transpose(grid, (1, 2, 0)))
plt.title("Mode collapse inspection")
plt.show()

# =========================================================
# ==== TASK 8: STABILIZATION ===============================
# =========================================================

# Label smoothing example
real_label = 0.9
fake_label = 0.1

# (You would retrain with these labels instead of 1 and 0)

# =========================================================
# ==== TASK 9: SIMPLE METRIC ===============================
# =========================================================

# Compare mean pixel intensity
real_pixels = real_batch.cpu().numpy().flatten()
fake_pixels = fake_batch.detach().cpu().numpy().flatten()

plt.hist(real_pixels, bins=50, alpha=0.5, label="Real")
plt.hist(fake_pixels, bins=50, alpha=0.5, label="Fake")
plt.legend()
plt.title("Pixel intensity distribution")
plt.show()

# =========================================================
# ==== DCGAN IMPLEMENTATION (FOR COMPARISON) ==============
# =========================================================

# =========================
# DCGAN GENERATOR
# =========================
class DCGenerator(nn.Module):
    """
    DCGAN Generator: uses ConvTranspose2d to generate images
    """
    def __init__(self, z_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


# =========================
# DCGAN DISCRIMINATOR
# =========================
class DCDiscriminator(nn.Module):
    """
    DCGAN Discriminator: uses Conv2d to classify images
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1)


# =========================
# INITIALIZE DCGAN
# =========================
dc_gen = DCGenerator(Z_DIM).to(device)
dc_disc = DCDiscriminator().to(device)

opt_gen_dc = optim.Adam(dc_gen.parameters(), lr=LR, betas=(0.5, 0.999))
opt_disc_dc = optim.Adam(dc_disc.parameters(), lr=LR, betas=(0.5, 0.999))

criterion = nn.BCELoss()

# =========================
# TRAIN DCGAN
# =========================
dc_g_losses = []
dc_d_losses = []

for epoch in range(EPOCHS):
    for real, _ in loader:
        real = real.to(device)
        batch_size = real.shape[0]

        # reshape noise for conv input
        noise = torch.randn(batch_size, Z_DIM, 1, 1).to(device)
        fake = dc_gen(noise)

        # === Train Discriminator ===
        real_labels = torch.ones(batch_size).to(device)
        fake_labels = torch.zeros(batch_size).to(device)

        disc_real = dc_disc(real)
        loss_real = criterion(disc_real, real_labels)

        disc_fake = dc_disc(fake.detach())
        loss_fake = criterion(disc_fake, fake_labels)

        loss_disc = (loss_real + loss_fake) / 2

        opt_disc_dc.zero_grad()
        loss_disc.backward()
        opt_disc_dc.step()

        # === Train Generator ===
        output = dc_disc(fake)
        loss_gen = criterion(output, real_labels)

        opt_gen_dc.zero_grad()
        loss_gen.backward()
        opt_gen_dc.step()

    dc_g_losses.append(loss_gen.item())
    dc_d_losses.append(loss_disc.item())

    print(f"[DCGAN] Epoch {epoch+1}/{EPOCHS} | G: {loss_gen:.4f} | D: {loss_disc:.4f}")


# =========================
# SHOW DCGAN RESULTS
# =========================
def show_dcgan_images(generator, title):
    noise = torch.randn(16, Z_DIM, 1, 1).to(device)
    fake = generator(noise).detach().cpu()
    grid = torchvision.utils.make_grid(fake, normalize=True)
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.title(title)
    plt.show()


show_dcgan_images(dc_gen, "DCGAN Generated Images")


# =========================
# COMPARE LOSSES
# =========================
plt.plot(g_losses, label="MLP Generator")
plt.plot(dc_g_losses, label="DCGAN Generator")
plt.legend()
plt.title("Generator Comparison")
plt.show()