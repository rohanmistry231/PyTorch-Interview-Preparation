import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np

# %% [1. Introduction to Experiment Management]
# Experiment management involves tracking, reproducibility, and hyperparameter search
# to ensure robust and repeatable machine learning experiments.

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

# Train/validation split
train_size = int(0.8 * len(X))
val_size = len(X) - train_size
train_dataset = CustomDataset(X[:train_size], y[:train_size])
val_dataset = CustomDataset(X[train_size:], y[train_size:])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
print("\nDummy Dataset (first 5 train samples):")
print("Features (X):\n", X[:5])
print("Labels (y):\n", y[:5])

# %% [3. Experiment Tracking]
# Simulate tracking metrics (e.g., loss, accuracy) as done in Weights & Biases.

def train_and_track(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    metrics = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = torch.sigmoid(model(features))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * features.size(0)
        train_loss /= len(train_loader.dataset)
        metrics['train_loss'].append(train_loss)
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = torch.sigmoid(model(features))
                loss = criterion(outputs, labels)
                val_loss += loss.item() * features.size(0)
                predictions = (outputs > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        val_loss /= len(val_loader.dataset)
        val_accuracy = correct / total
        metrics['val_loss'].append(val_loss)
        metrics['val_accuracy'].append(val_accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    
    return metrics

model = SimpleNet(input_size=2, hidden_size=8, output_size=1)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
print("\nExperiment Tracking:")
metrics = train_and_track(model, train_loader, val_loader, criterion, optimizer, num_epochs=5)
print("Final Metrics:", {k: [round(v, 4) for v in metrics[k]] for k in metrics})

# %% [4. Reproducible Research: Setting Random Seeds]
# Ensure reproducibility by setting seeds for all random operations.

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
model = SimpleNet(input_size=2, hidden_size=8, output_size=1)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("\nTraining with Fixed Seed:")
metrics_seed = train_and_track(model, train_loader, val_loader, criterion, optimizer, num_epochs=3)
print("Metrics with Seed 42:", {k: [round(v, 4) for v in metrics_seed[k]] for k in metrics_seed})

# %% [5. Reproducible Research: Versioning Code and Data]
# Simulate versioning by saving model and dataset states.

def save_experiment(model, dataset, metrics, path="experiment.pt"):
    torch.save({
        'model_state': model.state_dict(),
        'dataset_features': dataset.features,
        'dataset_labels': dataset.labels,
        'metrics': metrics
    }, path)

save_experiment(model, train_dataset, metrics_seed, path="experiment_v1.pt")
print("\nExperiment Saved to: experiment_v1.pt")

# Load and verify
loaded = torch.load("experiment_v1.pt")
loaded_model = SimpleNet(input_size=2, hidden_size=8, output_size=1)
loaded_model.load_state_dict(loaded['model_state'])
print("Loaded Model First Layer Weight (sample):", loaded_model.layer1.weight[0, :].detach().tolist())
print("Loaded Dataset Size:", len(loaded['dataset_features']))

# %% [6. Hyperparameter Search: Grid Search]
# Perform grid search over learning rate and hidden size.

param_grid = {
    'lr': [0.1, 0.01, 0.001],
    'hidden_size': [4, 8, 16]
}

print("\nGrid Search:")
best_val_acc = 0.0
best_params = None
for lr, hidden_size in itertools.product(param_grid['lr'], param_grid['hidden_size']):
    model = SimpleNet(input_size=2, hidden_size=hidden_size, output_size=1)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(3):  # Short training for demo
        model.train()
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = torch.sigmoid(model(features))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = torch.sigmoid(model(features))
                predictions = (outputs > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
    
    print(f"LR: {lr}, Hidden Size: {hidden_size}, Val Accuracy: {val_acc:.4f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_params = {'lr': lr, 'hidden_size': hidden_size}

print("Best Parameters:", best_params, "Best Val Accuracy:", best_val_acc)

# %% [7. Hyperparameter Search: Random Search]
# Perform random search over hyperparameters.

num_trials = 5
param_dist = {
    'lr': [0.0001, 0.001, 0.01, 0.1],
    'hidden_size': [4, 8, 16, 32]
}

print("\nRandom Search:")
best_val_acc = 0.0
best_params = None
for _ in range(num_trials):
    lr = random.choice(param_dist['lr'])
    hidden_size = random.choice(param_dist['hidden_size'])
    
    model = SimpleNet(input_size=2, hidden_size=hidden_size, output_size=1)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(3):
        model.train()
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = torch.sigmoid(model(features))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = torch.sigmoid(model(features))
                predictions = (outputs > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
    
    print(f"Trial: LR={lr}, Hidden Size={hidden_size}, Val Accuracy: {val_acc:.4f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_params = {'lr': lr, 'hidden_size': hidden_size}

print("Best Parameters:", best_params, "Best Val Accuracy:", best_val_acc)

# %% [8. Evaluation]
# Evaluate the best model from grid search.

model = SimpleNet(input_size=2, hidden_size=best_params['hidden_size'], output_size=1)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])

for epoch in range(5):
    model.train()
    for features, labels in train_loader:
        optimizer.zero_grad()
        outputs = torch.sigmoid(model(features))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for features, labels in val_loader:
        outputs = torch.sigmoid(model(features))
        predictions = (outputs > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
accuracy = correct / total
print("\nBest Model Validation Accuracy:", accuracy)