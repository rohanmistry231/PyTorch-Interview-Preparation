import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.autograd as autograd

# %% [1. Introduction to Custom Extensions]
# Custom extensions allow tailoring PyTorch for specific needs, e.g., custom autograd functions,
# optimizers, or C++/CUDA integrations.

# Simple neural network for testing
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

# %% [3. Custom Autograd Function]
# Define a custom autograd function for a non-standard operation (e.g., scaled ReLU).

class ScaledReLU(autograd.Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.save_for_backward(input)
        ctx.scale = scale
        return torch.relu(input) * scale
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        scale = ctx.scale
        grad_input = grad_output.clone()
        grad_input[input <= 0] = 0  # Gradient of ReLU
        grad_input *= scale
        return grad_input, None

# Use in a module
class CustomReLUNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, scale=2.0):
        super(CustomReLUNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.scale = scale
    
    def forward(self, x):
        x = self.layer1(x)
        x = ScaledReLU.apply(x, self.scale)
        x = self.layer2(x)
        return x

# Test custom ReLU
model = CustomReLUNet(input_size=2, hidden_size=8, output_size=1)
sample_input = torch.randn(5, 2, requires_grad=True)
output = model(sample_input)
output.mean().backward()
print("\nCustom ReLU Output Shape:", output.shape)
print("Gradient of Input:", sample_input.grad.shape)

# %% [4. Custom Optimizer]
# Implement a custom optimizer (e.g., simplified SGD with momentum).

class CustomSGD(optim.Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.9):
        defaults = dict(lr=lr, momentum=momentum)
        super(CustomSGD, self).__init__(params, defaults)
    
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(grad)
                p.data.add_(-lr * buf)

# Test custom optimizer
model = SimpleNet(input_size=2, hidden_size=8, output_size=1)
optimizer = CustomSGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.BCELoss()

print("\nTraining with Custom SGD (1 epoch):")
model.train()
for features, labels in dataloader:
    optimizer.zero_grad()
    outputs = torch.sigmoid(model(features))
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    print("Custom SGD Loss:", loss.item())
    break

# %% [5. Writing C++/CUDA Extensions (Simulated)]
# Simulate a C++/CUDA extension with a Python approximation.
# Example: Custom matrix multiplication with scaling.

class CustomMatMul(nn.Module):
    def __init__(self, scale=1.0):
        super(CustomMatMul, self).__init__()
        self.scale = scale
    
    def forward(self, x, weight):
        # Simulate custom C++/CUDA: Scaled matrix multiplication
        return torch.matmul(x, weight.t()) * self.scale

# Integrate into a model
class CustomMatMulNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomMatMulNet, self).__init__()
        self.weight1 = nn.Parameter(torch.randn(hidden_size, input_size))
        self.custom_matmul = CustomMatMul(scale=2.0)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.custom_matmul(x, self.weight1)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# Test custom matmul model
model = CustomMatMulNet(input_size=2, hidden_size=8, output_size=1)
output = model(sample_input)
print("\nCustom MatMul Model Output Shape:", output.shape)

# %% [6. Training with Custom Extensions]
# Train a model with custom ReLU and optimizer.

model = CustomReLUNet(input_size=2, hidden_size=8, output_size=1, scale=2.0)
criterion = nn.BCELoss()
optimizer = CustomSGD(model.parameters(), lr=0.01, momentum=0.9)
num_epochs = 5

print("\nTraining with Custom ReLU and SGD:")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for features, labels in dataloader:
        optimizer.zero_grad()
        outputs = torch.sigmoid(model(features))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * features.size(0)
    epoch_loss = running_loss / len(dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# %% [7. Debugging Custom Extensions]
# Inspect gradients and outputs of custom autograd function.

model = CustomReLUNet(input_size=2, hidden_size=8, output_size=1, scale=2.0)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("\nDebugging Custom ReLU Gradients:")
for features, labels in dataloader:
    optimizer.zero_grad()
    outputs = torch.sigmoid(model(features))
    loss = criterion(outputs, labels)
    loss.backward()
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name} Gradient Norm: {param.grad.norm().item():.4f}")
    print("Sample Output:", outputs[:2].detach().flatten().tolist())
    break

# %% [8. Evaluation]
# Evaluate the custom model.

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for features, labels in dataloader:
        outputs = torch.sigmoid(model(features))
        predictions = (outputs > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
accuracy = correct / total
print("\nCustom Model Accuracy:", accuracy)