import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os

# %% [1. Introduction to Training a Neural Network]
# Training involves a forward pass, loss computation, backward pass, and parameter updates.
# Key components: model, loss function, optimizer, and training/evaluation loops.

# Define a simple neural network
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
# Synthetic dataset: 100 samples, 2 features, 1 binary label (0 or 1).
torch.manual_seed(42)
X = torch.randn(100, 2)  # 100 samples, 2 features
y = (X[:, 0] + X[:, 1] > 0).float().reshape(-1, 1)  # Label: 1 if sum > 0, else 0

# Custom dataset
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

dataset = CustomDataset(X, y)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
print("\nDummy Dataset (first 5 samples):")
print("Features (X):\n", X[:5])
print("Labels (y):\n", y[:5])

# %% [3. Training Loop]
# Implement a training loop with forward pass, loss computation, and optimization.

model = SimpleNet(input_size=2, hidden_size=8, output_size=1)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 10

print("\nTraining Loop:")
for epoch in range(num_epochs):
    model.train()  # Training mode
    running_loss = 0.0
    for features, labels in train_loader:
        optimizer.zero_grad()  # Clear gradients
        outputs = torch.sigmoid(model(features))  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update parameters
        running_loss += loss.item() * features.size(0)
    epoch_loss = running_loss / len(dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# %% [4. Evaluation Loop]
# Evaluate the model on the training data (for simplicity).

model.eval()  # Evaluation mode
correct = 0
total = 0
with torch.no_grad():
    for features, labels in train_loader:
        outputs = torchesigmoid(model(features))
        predictions = (outputs > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
accuracy = correct / total
print("\nTraining Accuracy:", accuracy)

# %% [5. Model Checkpointing]
# Save and load model weights for later use.

# Save model
checkpoint_path = "model_checkpoint.pt"
torch.save(model.state_dict(), checkpoint_path)
print("\nModel saved to:", checkpoint_path)

# Load model
loaded_model = SimpleNet(input_size=2, hidden_size=8, output_size=1)
loaded_model.load_state_dict(torch.load(checkpoint_path))
loaded_model.eval()
print("Model loaded from:", checkpoint_path)

# Verify loaded model
with torch.no_grad():
    outputs = torch.sigmoid(loaded_model(X[:5]))
    print("Predictions from loaded model (first 5):\n", outputs.flatten().tolist())

# %% [6. Training on GPU]
# Move model and data to GPU if available.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nDevice:", device)

# Move model to device
model = SimpleNet(input_size=2, hidden_size=8, output_size=1).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop with device
for epoch in range(3):  # Short loop for demonstration
    model.train()
    running_loss = 0.0
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)  # Move to device
        optimizer.zero_grad()
        outputs = torch.sigmoid(model(features))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * features.size(0)
    epoch_loss = running_loss / len(dataset)
    print(f"Epoch {epoch+1}/3 (GPU), Loss: {epoch_loss:.4f}")

# %% [7. Monitoring Training]
# Track loss and accuracy during training.

model = SimpleNet(input_size=2, hidden_size=8, output_size=1)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("\nTraining with Monitoring:")
for epoch in range(5):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for features, labels in train_loader:
        optimizer.zero_grad()
        outputs = torch.sigmoid(model(features))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * features.size(0)
        predictions = (outputs > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    epoch_loss = running_loss / len(dataset)
    epoch_acc = correct / total
    print(f"Epoch {epoch+1}/5, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

# %% [8. Visualization (Simulated with Print)]
# Simulate visualization by printing loss trends (in practice, use Matplotlib/TensorBoard).

model = SimpleNet(input_size=2, hidden_size=8, output_size=1)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
losses = []

print("\nTraining with Loss Tracking:")
for epoch in range(5):
    model.train()
    running_loss = 0.0
    for features, labels in train_loader:
        optimizer.zero_grad()
        outputs = torch.sigmoid(model(features))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * features.size(0)
    epoch_loss = running_loss / len(dataset)
    losses.append(epoch_loss)
    print(f"Epoch {epoch+1}/5, Loss: {epoch_loss:.4f}")

print("\nLoss Trend:", [round(l, 4) for l in losses])