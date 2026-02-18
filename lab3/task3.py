import torch
import torch.nn as nn

class BaselineCNN(nn.Module):
    def __init__(self, num_classes):
        super(BaselineCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Binary model
binary_model = BaselineCNN(num_classes=2)

# Multiclass model
multi_model = BaselineCNN(num_classes=3)

# Print model summaries
print("Binary Model Architecture:")
print(binary_model)

print("\nMulticlass Model Architecture:")
print(multi_model)

# Count trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("\nBinary Model Trainable Parameters:", count_parameters(binary_model))
print("Multiclass Model Trainable Parameters:", count_parameters(multi_model))
