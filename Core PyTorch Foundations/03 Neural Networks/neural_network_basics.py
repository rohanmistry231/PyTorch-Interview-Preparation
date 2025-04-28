import torch
import torch.nn as nn
import torch.optim as optim

# %% [1. Introduction to torch.nn.Module]
# torch.nn.Module is the base class for all neural networks in PyTorch.
# Subclass it to define custom models with layers, forward pass, and parameters.

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

# Create and inspect model
model = SimpleNet(input_size=2, hidden_size=4, output_size=1)
print("Model Architecture:\n", model)
print("\nModel Parameters:")
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# %% [2. Dummy Dataset]
# Synthetic dataset for binary classification: 100 samples, 2 features, 1 label (0 or 1).
torch.manual_seed(42)
X = torch.randn(100, 2)  # 100 samples, 2 features
y = (X[:, 0] + X[:, 1] > 0).float().reshape(-1, 1)  # Label: 1 if sum > 0, else 0

print("\nDummy Dataset (first 5 samples):")
print("Features (X):\n", X[:5])
print("Labels (y):\n", y[:5])

# %% [3. Layers and Activations]
# Common layers and activations in torch.nn for building neural networks.

# Define a small network with various layers
class DemoNet(nn.Module):
    def __init__(self):
        super(DemoNet, self).__init__()
        self.linear = nn.Linear(2, 4)
        self.conv = nn.Conv2d(1, 2, kernel_size=2)  # For 2D input
        self.batch_norm = nn.BatchNorm1d(4)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.linear(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.sigmoid(x)
        return x

demo_model = DemoNet()
print("\nDemoNet Architecture:\n", demo_model)

# Test with a sample input
sample_input = X[:1]  # One sample
output = demo_model(sample_input)
print("\nOutput of DemoNet for one sample:\n", output)

# %% [4. Loss Functions]
# Common loss functions for different tasks.

# Binary cross-entropy loss for classification
criterion_bce = nn.BCELoss()
y_pred = torch.sigmoid(model(X))  # Sigmoid for binary classification
loss_bce = criterion_bce(y_pred, y)
print("\nBinary Cross-Entropy Loss:", loss_bce.item())

# Mean squared error for regression
criterion_mse = nn.MSELoss()
loss_mse = criterion_mse(y_pred, y)
print("Mean Squared Error Loss:", loss_mse.item())

# Cross-entropy loss (for multi-class, requires raw logits)
y_multi = y.long().squeeze()  # Convert to class indices
model_multi = SimpleNet(2, 4, 2)  # 2 output classes
criterion_ce = nn.CrossEntropyLoss()
logits = model_multi(X)
loss_ce = criterion_ce(logits, y_multi)
print("Cross-Entropy Loss:", loss_ce.item())

# %% [5. Optimizers]
# Optimizers update model parameters based on gradients.

# SGD optimizer
optimizer_sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
print("\nSGD Optimizer:", optimizer_sgd)

# Adam optimizer
optimizer_adam = optim.Adam(model.parameters(), lr=0.001)
print("Adam Optimizer:", optimizer_adam)

# Simulate one optimization step
model.zero_grad()  # Clear gradients
loss_bce.backward()  # Compute gradients
optimizer_adam.step()  # Update parameters
print("\nAfter one Adam step, first layer weight (sample):\n", model.layer1.weight[0, :])

# %% [6. Learning Rate Scheduling]
# Schedulers adjust the learning rate during training.

# StepLR: Reduce LR by a factor every few epochs
scheduler = optim.lr_scheduler.StepLR(optimizer_adam, step_size=2, gamma=0.1)
initial_lr = optimizer_adam.param_groups[0]['lr']
print("\nInitial Learning Rate:", initial_lr)

# Simulate 4 epochs
for epoch in range(4):
    scheduler.step()
    current_lr = optimizer_adam.param_groups[0]['lr']
    print(f"Epoch {epoch+1}, Learning Rate: {current_lr}")

# %% [7. Training Loop Example]
# A simple training loop for the binary classification task.

model = SimpleNet(2, 4, 1)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()
num_epochs = 5

print("\nTraining Loop:")
for epoch in range(num_epochs):
    model.train()  # Set to training mode
    optimizer.zero_grad()  # Clear gradients
    y_pred = torch.sigmoid(model(X))  # Forward pass
    loss = criterion(y_pred, y)  # Compute loss
    loss.backward()  # Compute gradients
    optimizer.step()  # Update parameters
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# %% [8. Evaluation]
# Evaluate the model on the training data (for simplicity).

model.eval()  # Set to evaluation mode
with torch.no_grad():
    y_pred = torch.sigmoid(model(X))
    predictions = (y_pred > 0.5).float()  # Threshold at 0.5
    accuracy = (predictions == y).float().mean()
print("\nTraining Accuracy:", accuracy.item())

# Example predictions
print("Sample Predictions (first 5):\n", predictions[:5].flatten().tolist())
print("True Labels (first 5):\n", y[:5].flatten().tolist())