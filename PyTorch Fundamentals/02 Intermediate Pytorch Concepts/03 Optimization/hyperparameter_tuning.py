import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import itertools

# %% [1. Introduction to Hyperparameter Tuning]
# Hyperparameters (e.g., learning rate, batch size) affect model performance.
# Tuning involves experimenting with different values to optimize results.

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
print("\nDummy Dataset (first 5 samples):")
print("Features (X):\n", X[:5])
print("Labels (y):\n", y[:5])

# %% [3. Learning Rate Tuning]
# Test different learning rates to find the best one.

learning_rates = [0.1, 0.01, 0.001]
num_epochs = 5

print("\nLearning Rate Tuning:")
for lr in learning_rates:
    model = SimpleNet(input_size=2, hidden_size=8, output_size=1)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
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
    print(f"Learning Rate: {lr}, Final Loss: {epoch_loss:.4f}")

# %% [4. Batch Size Selection]
# Test different batch sizes to balance speed and stability.

batch_sizes = [8, 16, 32]
print("\nBatch Size Tuning:")
for batch_size in batch_sizes:
    model = SimpleNet(input_size=2, hidden_size=8, output_size=1)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
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
    print(f"Batch Size: {batch_size}, Final Loss: {epoch_loss:.4f}")

# %% [5. Optimizer Configuration]
# Compare different optimizers.

optimizers = {
    'SGD': optim.SGD,
    'Adam': optim.Adam,
    'RMSprop': optim.RMSprop
}
print("\nOptimizer Tuning:")
for opt_name, opt_class in optimizers.items():
    model = SimpleNet(input_size=2, hidden_size=8, output_size=1)
    criterion = nn.BCELoss()
    optimizer = opt_class(model.parameters(), lr=0.01)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
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
    print(f"Optimizer: {opt_name}, Final Loss: {epoch_loss:.4f}")

# %% [6. Regularization: Weight Decay]
# Apply weight decay to prevent overfitting.

weight_decays = [0.0, 0.001, 0.01]
print("\nWeight Decay Tuning:")
for wd in weight_decays:
    model = SimpleNet(input_size=2, hidden_size=8, output_size=1)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=wd)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
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
    print(f"Weight Decay: {wd}, Final Loss: {epoch_loss:.4f}")

# %% [7. Regularization: Dropout]
# Add dropout to the model to reduce overfitting.

dropout_rates = [0.0, 0.2, 0.5]
class DropoutNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate):
        super(DropoutNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

print("\nDropout Tuning:")
for dropout_rate in dropout_rates:
    model = DropoutNet(input_size=2, hidden_size=8, output_size=1, dropout_rate=dropout_rate)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
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
    print(f"Dropout Rate: {dropout_rate}, Final Loss: {epoch_loss:.4f}")

# %% [8. Early Stopping]
# Stop training when validation loss stops improving.

class EarlyStopping:
    def __init__(self, patience=2, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# Simulate train/validation split (80/20)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

model = SimpleNet(input_size=2, hidden_size=8, output_size=1)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
early_stopping = EarlyStopping(patience=2)

print("\nTraining with Early Stopping:")
for epoch in range(10):
    model.train()
    train_loss = 0.0
    for features, labels in train_loader:
        optimizer.zero_grad()
        outputs = torch.sigmoid(model(features))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * features.size(0)
    train_loss /= len(train_dataset)
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for features, labels in val_loader:
            outputs = torch.sigmoid(model(features))
            loss = criterion(outputs, labels)
            val_loss += loss.item() * features.size(0)
    val_loss /= len(val_dataset)
    
    print(f"Epoch {epoch+1}/10, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print("Early stopping triggered!")
        break