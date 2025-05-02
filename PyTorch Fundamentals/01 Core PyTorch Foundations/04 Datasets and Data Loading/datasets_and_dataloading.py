import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np

# %% [1. Introduction to torch.utils.data]
# torch.utils.data provides tools for data loading: Dataset for data storage and DataLoader for batching.
# Dataset defines how data is accessed; DataLoader iterates over it efficiently.

# Example: Creating a simple tensor dataset
tensor_dataset = torch.utils.data.TensorDataset(torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]))
print("Tensor Dataset Example:", list(tensor_dataset))  # List of (input, label) tuples

# %% [2. Dummy Dataset]
# Synthetic dataset: 100 samples, 2 features (e.g., height, weight), 1 binary label (e.g., healthy/unhealthy).
torch.manual_seed(42)
X = torch.randn(100, 2)  # 100 samples, 2 features
y = (X[:, 0] + X[:, 1] > 0).float().reshape(-1, 1)  # Label: 1 if sum > 0, else 0

print("\nDummy Dataset (first 5 samples):")
print("Features (X):\n", X[:5])
print("Labels (y):\n", y[:5])

# %% [3. Built-in Datasets]
# torchvision.datasets provides access to standard datasets like MNIST.

from torchvision.datasets import MNIST
# Load MNIST (downloads if not present)
mnist_dataset = MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
print("\nMNIST Dataset:")
print("Number of samples:", len(mnist_dataset))
print("Sample shape:", mnist_dataset[0][0].shape)  # Image: (1, 28, 28)
print("Label of first sample:", mnist_dataset[0][1])  # Label: 0-9

# Access one sample
image, label = mnist_dataset[0]
print("First MNIST image shape:", image.shape)

# %% [4. Custom Dataset]
# Create a custom dataset by subclassing torch.utils.data.Dataset.

class CustomDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        sample = self.features[idx], self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

# Instantiate custom dataset
custom_dataset = CustomDataset(X, y)
print("\nCustom Dataset:")
print("Length:", len(custom_dataset))
print("First sample:", custom_dataset[0])

# %% [5. DataLoader]
# DataLoader provides batching, shuffling, and parallel loading.

# Create DataLoader for custom dataset
dataloader = DataLoader(custom_dataset, batch_size=4, shuffle=True, num_workers=2)
print("\nDataLoader with batch_size=4:")
for batch_features, batch_labels in dataloader:
    print("Batch Features Shape:", batch_features.shape)  # (4, 2)
    print("Batch Labels Shape:", batch_labels.shape)  # (4, 1)
    break  # Print only first batch

# DataLoader for MNIST
mnist_loader = DataLoader(mnist_dataset, batch_size=16, shuffle=True)
print("\nMNIST DataLoader (first batch):")
for images, labels in mnist_loader:
    print("Batch Images Shape:", images.shape)  # (16, 1, 28, 28)
    print("Batch Labels Shape:", labels.shape)  # (16,)
    break

# %% [6. Data Preprocessing with Transforms]
# torchvision.transforms applies preprocessing (e.g., normalization, augmentation).

# Define transforms for MNIST
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL Image to tensor
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize with MNIST mean and std
])

# Reload MNIST with transforms
mnist_transformed = MNIST(root='./data', train=True, download=True, transform=transform)
print("\nMNIST with Transforms:")
image, label = mnist_transformed[0]
print("Transformed Image Shape:", image.shape)
print("Transformed Image Mean:", image.mean().item())

# Custom transform for dummy dataset
def custom_transform(sample):
    features, label = sample
    # Example: Scale features
    features = features * 2
    return features, label

custom_dataset_transformed = CustomDataset(X, y, transform=custom_transform)
print("\nCustom Dataset with Transform (first sample):")
print(custom_dataset_transformed[0])

# %% [7. Custom Collate Function]
# Collate function defines how samples are combined into batches.

def custom_collate_fn(batch):
    # Stack features and labels, convert labels to long
    features = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return features, labels

dataloader_custom_collate = DataLoader(custom_dataset, batch_size=4, collate_fn=custom_collate_fn)
print("\nDataLoader with Custom Collate:")
for features, labels in dataloader_custom_collate:
    print("Batch Features Shape:", features.shape)
    print("Batch Labels Shape:", labels.shape)
    print("Labels dtype:", labels.dtype)
    break

# %% [8. Handling Large Datasets]
# Techniques for large datasets: streaming or disk-based loading.

# Simulate streaming by loading subsets
subset_indices = list(range(20))  # First 20 samples
subset_dataset = torch.utils.data.Subset(custom_dataset, subset_indices)
subset_loader = DataLoader(subset_dataset, batch_size=5)
print("\nSubset DataLoader (20 samples):")
for features, labels in subset_loader:
    print("Batch Features Shape:", features.shape)
    break

# Example: Disk-based loading (simulated with in-memory data)
# Save dummy dataset to disk
torch.save({'features': X, 'labels': y}, 'dummy_dataset.pt')

# Load from disk
loaded_data = torch.load('dummy_dataset.pt')
loaded_dataset = CustomDataset(loaded_data['features'], loaded_data['labels'])
print("\nLoaded Dataset from Disk:")
print("First sample:", loaded_dataset[0])