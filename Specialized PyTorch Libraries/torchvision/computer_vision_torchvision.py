import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

# %% [1. Introduction to torchvision]
# torchvision provides datasets, pretrained models, transforms, and utilities for computer vision tasks.

# Check torchvision version
print("torchvision version:", torchvision.__version__)

# %% [2. Datasets (torchvision.datasets)]
# Load and explore the MNIST dataset.

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST dataset
mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)

print("\nMNIST Dataset:")
print("Training samples:", len(mnist_train))
print("Test samples:", len(mnist_test))
print("Sample image shape:", mnist_train[0][0].shape)  # (1, 28, 28)
print("Sample label:", mnist_train[0][1])

# %% [3. Pretrained Models (torchvision.models)]
# Use a pretrained ResNet18 for transfer learning.

resnet = models.resnet18(pretrained=True)
resnet.fc = nn.Linear(resnet.fc.in_features, 10)  # Modify for 10 MNIST classes
print("\nModified ResNet18:")
print(resnet.fc)

# Test with a dummy input
dummy_input = torch.randn(4, 3, 224, 224)
output = resnet(dummy_input)
print("ResNet Output Shape:", output.shape)  # Expected: (4, 10)

# %% [4. Transforms (torchvision.transforms)]
# Apply data augmentation and preprocessing.

augmented_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST with augmentation
mnist_train_aug = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=augmented_transform)
aug_loader = DataLoader(mnist_train_aug, batch_size=64, shuffle=True)

print("\nAugmented MNIST Dataset:")
image, label = mnist_train_aug[0]
print("Augmented Image Shape:", image.shape)
print("Augmented Image Mean:", image.mean().item())

# %% [5. Utilities for Image Processing]
# Use torchvision.utils for image processing tasks, e.g., saving images.

from torchvision.utils import save_image

# Save a batch of images
images, labels = next(iter(train_loader))
save_image(images[:4], "mnist_samples.png", nrow=2, normalize=True)
print("\nSaved 4 MNIST images to: mnist_samples.png")

# %% [6. Training a Simple CNN on MNIST]
# Define and train a CNN using torchvision datasets.

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

cnn = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.001)
num_epochs = 3

print("\nTraining Simple CNN on MNIST:")
for epoch in range(num_epochs):
    cnn.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# %% [7. Fine-tuning Pretrained ResNet]
# Fine-tune ResNet18 on MNIST (simplified with grayscale-to-RGB conversion).

# Transform to match ResNet input (3 channels, 224x224)
resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # Convert 1 channel to 3
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

mnist_train_resnet = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=resnet_transform)
train_loader_resnet = DataLoader(mnist_train_resnet, batch_size=32, shuffle=True)

resnet = models.resnet18(pretrained=True)
resnet.fc = nn.Linear(resnet.fc.in_features, 10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.fc.parameters(), lr=0.001)  # Train only final layer

print("\nFine-tuning ResNet18 on MNIST:")
for epoch in range(2):  # Short training for demo
    resnet.train()
    running_loss = 0.0
    for images, labels in train_loader_resnet:
        optimizer.zero_grad()
        outputs = resnet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(train_loader_resnet.dataset)
    print(f"Epoch {epoch+1}/2, Loss: {epoch_loss:.4f}")

# %% [8. Evaluation]
# Evaluate the CNN on the test set.

cnn.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = cnn(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = correct / total
print("\nCNN Test Accuracy:", accuracy)