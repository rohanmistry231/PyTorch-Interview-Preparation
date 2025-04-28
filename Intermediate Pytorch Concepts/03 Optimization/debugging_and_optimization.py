import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.cuda.amp as amp

# %% [1. Introduction to Debugging and Optimization]
# Debugging involves identifying issues like NaN gradients or vanishing gradients.
# Optimization techniques improve training efficiency (e.g., mixed precision).

# Simple neural network
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# %% [2. Dummy Dataset]
# Synthetic dataset: 100 samples, 2 features, 1 binary label.
torch.manual_seed(42)
X = torch.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).float().reshape(-1, 1)

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

dataset = CustomDataset(X, y)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
print("\nDummy Dataset (first 5 samples):")
print("Features (X):\n", X[:5])
print("Labels (y):\n", y[:5])

# %% [3. Debugging: NaN/Inf Gradients]
# Simulate and detect NaN/Inf gradients.

model = SimpleNet(input_size=2, hidden_size=8, output_size=1)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1000.0)  # High LR to cause issues

print("\nChecking for NaN/Inf Gradients:")
for features, labels in dataloader:
    optimizer.zero_grad()
    outputs = torch.sigmoid(model(features))
    loss = criterion(outputs, labels)
    loss.backward()
    
    # Check for NaN/Inf in gradients
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print(f"NaN/Inf detected in gradients of {name}")
    break

# Fix: Use gradient clipping
model = SimpleNet(input_size=2, hidden_size=8, output_size=1)
optimizer = optim.Adam(model.parameters(), lr=1000.0)
for features, labels in dataloader:
    optimizer.zero_grad()
    outputs = torch.sigmoid(model(features))
    loss = criterion(outputs, labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    print("Applied gradient clipping, loss:", loss.item())
    break

# %% [4. Debugging: Vanishing/Exploding Gradients]
# Inspect gradient norms to detect vanishing/exploding gradients.

model = SimpleNet(input_size=2, hidden_size=8, output_size=1)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

print("\nGradient Norms:")
for features, labels in dataloader:
    optimizer.zero_grad()
    outputs = torch.sigmoid(model(features))
    loss = criterion(outputs, labels)
    loss.backward()
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"{name} gradient norm: {grad_norm:.4f}")
    break

# Simulate exploding gradients with high loss scaling
loss = loss * 1000
loss.backward()
print("\nAfter scaling loss (exploding gradients):")
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        print(f"{name} gradient norm: {grad_norm:.4f}")
break

# %% [5. Mixed Precision Training]
# Use torch.cuda.amp for faster training with lower memory usage.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNet(input_size=2, hidden_size=8, output_size=1).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scaler = amp.GradScaler(enabled=device.type == "cuda")

print("\nMixed Precision Training (1 epoch):")
model.train()
running_loss = 0.0
for features, labels in dataloader:
    features, labels = features.to(device), labels.to(device)
    optimizer.zero_grad()
    with amp.autocast(enabled=device.type == "cuda"):
        outputs = torch.sigmoid(model(features))
        loss = criterion(outputs, labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    running_loss += loss.item() * features.size(0)
epoch_loss = running_loss / len(dataset)
print(f"Epoch Loss (Mixed Precision): {epoch_loss:.4f}")

# %% [6. Model Optimization: Pruning]
# Apply pruning to reduce model size.

import torch.nn.utils.prune as prune

model = SimpleNet(input_size=2, hidden_size=8, output_size=1)
print("\nBefore Pruning (Layer1 Weights):")
print(model.layer1.weight.shape)

# Apply 50% unstructured pruning to layer1 weights
prune.random_unstructured(model.layer1, name="weight", amount=0.5)
print("After Pruning (Layer1 Weights Mask):")
print(model.layer1.weight_mask.shape)  # Mask indicates pruned weights
print("Non-zero Weights:", torch.count_nonzero(model.layer1.weight).item())

# Make pruning permanent
prune.remove(model.layer1, "weight")
print("After Making Pruning Permanent (Layer1 Weights):")
print(model.layer1.weight.shape)

# %% [7. Model Optimization: Quantization]
# Simulate post-training quantization for reduced model size.

model = SimpleNet(input_size=2, hidden_size=8, output_size=1).eval()
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
quantized_model = torch.quantization.prepare(model, inplace=False)
quantized_model = torch.quantization.convert(quantized_model, inplace=False)

print("\nQuantized Model:")
print(quantized_model)

# Test quantized model
sample_input = torch.randn(5, 2)
output = torch.sigmoid(quantized_model(sample_input))
print("Quantized Model Output Shape:", output.shape)

# %% [8. Memory Management]
# Manage GPU memory and use gradient checkpointing.

# Clear GPU memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("\nGPU Memory Cleared")

# Gradient checkpointing (trade computation for memory)
from torch.utils.checkpoint import checkpoint_sequential

class DeepNet(nn.Module):
    def __init__(self):
        super(DeepNet, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(2, 8) for _ in range(5)])
        self.relu = nn.ReLU()
        self.final = nn.Linear(8, 1)
    
    def forward(self, x):
        for layer in self.layers:
            x = checkpoint_sequential(nn.Sequential(layer, self.relu), segments=1, input=x)
        x = self.final(x)
        return x

# Test with gradient checkpointing
deep_model = DeepNet().to(device)
deep_model.train()
optimizer = optim.Adam(deep_model.parameters(), lr=0.01)
for features, labels in dataloader:
    features, labels = features.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = torch.sigmoid(deep_model(features))
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    print("\nGradient Checkpointing Loss:", loss.item())
    break