import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# %% [1. Writing Custom Layers]
# Create custom layers by subclassing nn.Module.

class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(CustomLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
    
    def forward(self, x):
        return torch.matmul(x, self.weight.t()) + self.bias

# Test custom layer
custom_layer = CustomLinear(in_features=2, out_features=4)
sample_input = torch.randn(5, 2)
output = custom_layer(sample_input)
print("Custom Linear Layer Output Shape:", output.shape)  # Expected: (5, 4)
print("Custom Layer Parameters:")
for param in custom_layer.parameters():
    print(param.shape)

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

# %% [3. Custom Loss Functions]
# Define a custom loss function for a specific task.

class CustomBCELoss(nn.Module):
    def __init__(self):
        super(CustomBCELoss, self).__init__()
    
    def forward(self, input, target):
        # Custom BCE: -mean(target * log(input) + (1-target) * log(1-input))
        input = torch.clamp(input, min=1e-7, max=1-1e-7)  # Avoid log(0)
        loss = -torch.mean(target * torch.log(input) + (1 - target) * torch.log(1 - input))
        return loss

# Test custom loss
custom_loss_fn = CustomBCELoss()
sample_pred = torch.sigmoid(torch.randn(5, 1))
sample_target = torch.tensor([[1.], [0.], [1.], [0.], [1.]])
loss = custom_loss_fn(sample_pred, sample_target)
print("\nCustom BCE Loss:", loss.item())

# %% [4. Dynamic Computation Graphs]
# Build models with conditional logic in the forward pass.

class DynamicNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DynamicNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x, use_extra_layer=False):
        x = self.layer1(x)
        x = self.relu(x)
        if use_extra_layer:
            x = self.layer2(x)
            x = self.relu(x)
            x = self.layer2(x)  # Apply layer2 again
        else:
            x = self.layer2(x)
        return x

# Test dynamic network
dynamic_net = DynamicNet(input_size=2, hidden_size=4, output_size=1)
output_normal = dynamic_net(sample_input)
output_extra = dynamic_net(sample_input, use_extra_layer=True)
print("\nDynamicNet Output (Normal):", output_normal.shape)  # Expected: (5, 1)
print("DynamicNet Output (Extra Layer):", output_extra.shape)  # Expected: (5, 1)

# %% [5. Model with Custom Layer]
# Build a model using the custom linear layer.

class CustomModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomModel, self).__init__()
        self.layer1 = CustomLinear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = CustomLinear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# Test custom model
custom_model = CustomModel(input_size=2, hidden_size=4, output_size=1)
output = custom_model(sample_input)
print("\nCustom Model Output Shape:", output.shape)  # Expected: (5, 1)

# %% [6. Training with Custom Model and Loss]
# Train the custom model with the custom loss function.

custom_model = CustomModel(input_size=2, hidden_size=8, output_size=1)
criterion = CustomBCELoss()
optimizer = torch.optim.Adam(custom_model.parameters(), lr=0.01)
num_epochs = 5

print("\nTraining Custom Model:")
for epoch in range(num_epochs):
    custom_model.train()
    running_loss = 0.0
    for features, labels in dataloader:
        optimizer.zero_grad()
        outputs = torch.sigmoid(custom_model(features))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * features.size(0)
    epoch_loss = running_loss / len(dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# %% [7. Model Debugging: Gradient Flow]
# Inspect gradient flow to debug training issues.

# Train one step and check gradients
custom_model.zero_grad()
sample_batch = next(iter(dataloader))
features, labels = sample_batch
outputs = torch.sigmoid(custom_model(features))
loss = criterion(outputs, labels)
loss.backward()

print("\nGradient Norms:")
for name, param in custom_model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm().item():.4f}")

# %% [8. Model Debugging: Intermediate Outputs]
# Inspect intermediate outputs to understand model behavior.

class DebugNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DebugNet, self).__init__()
        self.layer1 = CustomLinear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = CustomLinear(hidden_size, output_size)
    
    def forward(self, x):
        x1 = self.layer1(x)
        print("After Layer1:", x1[:2])  # Print first 2 samples
        x2 = self.relu(x1)
        print("After ReLU:", x2[:2])
        x3 = self.layer2(x2)
        print("After Layer2:", x3[:2])
        return x3

# Test debug model
debug_model = DebugNet(input_size=2, hidden_size=4, output_size=1)
output = debug_model(sample_input[:2])  # Small batch for clarity
print("\nDebug Model Output Shape:", output.shape)