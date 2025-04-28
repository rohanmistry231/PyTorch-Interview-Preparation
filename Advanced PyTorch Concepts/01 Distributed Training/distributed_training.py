import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# %% [1. Introduction to Distributed Training]
# Distributed training splits computation across multiple GPUs or nodes to speed up training.
# PyTorch supports DataParallel (single node) and DistributedDataParallel (multi-node).

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
print("\nDummy Dataset (first 5 samples):")
print("Features (X):\n", X[:5])
print("Labels (y):\n", y[:5])

# %% [3. Data Parallelism (DataParallel)]
# DataParallel splits a batch across multiple GPUs on a single node.

def train_data_parallel():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNet(input_size=2, hidden_size=8, output_size=1).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)  # Wrap model for multi-GPU
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    print("\nTraining with DataParallel (1 epoch):")
    model.train()
    running_loss = 0.0
    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)
        optimizer zeroes_grad()
        outputs = torch.sigmoid(model(features))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * features.size(0)
    epoch_loss = running_loss / len(dataset)
    print(f"DataParallel Loss: {epoch_loss:.4f}")

if torch.cuda.is_available():
    train_data_parallel()
else:
    print("\nDataParallel skipped: GPU not available")

# %% [4. Distributed Data Parallel (DDP) Setup]
# DDP is used for multi-GPU or multi-node training, requiring process initialization.

def setup_ddp(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

# %% [5. DDP Training Function]
# Train with DDP, where each process handles a portion of the data.

def train_ddp(rank, world_size):
    setup_ddp(rank, world_size)
    model = SimpleNet(input_size=2, hidden_size=8, output_size=1).to(rank)
    model = DDP(model, device_ids=[rank])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Create distributed sampler
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=16, sampler=sampler)
    
    print(f"\nTraining with DDP (Rank {rank}, 1 epoch):")
    model.train()
    running_loss = 0.0
    for features, labels in dataloader:
        features, labels = features.to(rank), labels.to(rank)
        optimizer.zero_grad()
        outputs = torch.sigmoid(model(features))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * features.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    print(f"Rank {rank}, DDP Loss: {epoch_loss:.4f}")
    
    cleanup_ddp()

# %% [6. Running DDP]
# Spawn processes for DDP (simulates multi-GPU training).

def run_ddp(world_size):
    mp.spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)

if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    world_size = torch.cuda.device_count()
    print(f"\nRunning DDP with {world_size} GPUs")
    run_ddp(world_size)
else:
    print("\nDDP skipped: Requires multiple GPUs")

# %% [7. Multi-Node Training (Simulated)]
# Simulate multi-node setup by running DDP on a single node.

# Note: True multi-node requires environment variables (e.g., MASTER_ADDR, MASTER_PORT)
# This is a simplified single-node simulation
def simulate_multi_node(rank, world_size):
    setup_ddp(rank, world_size)
    model = SimpleNet(input_size=2, hidden_size=8, output_size=1).to(rank)
    model = DDP(model, device_ids=[rank])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=16, sampler=sampler)
    
    print(f"\nSimulated Multi-Node Training (Rank {rank}):")
    model.train()
    for features, labels in dataloader:
        features, labels = features.to(rank), labels.to(rank)
        optimizer.zero_grad()
        outputs = torch.sigmoid(model(features))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f"Rank {rank}, Loss: {loss.item():.4f}")
        break
    
    cleanup_ddp()

if torch.cuda.is_available():
    world_size = min(2, torch.cuda.device_count())  # Simulate 2 nodes
    print(f"\nSimulating Multi-Node with {world_size} processes")
    mp.spawn(simulate_multi_node, args=(world_size,), nprocs=world_size, join=True)
else:
    print("\nMulti-Node simulation skipped: GPU not available")

# %% [8. Evaluation]
# Evaluate the model after training (using DataParallel for simplicity).

model = SimpleNet(input_size=2, hidden_size=8, output_size=1)
if torch.cuda.is_available():
    model = nn.DataParallel(model).cuda()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Train for 1 epoch
model.train()
for features, labels in dataloader:
    features, labels = features.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = torch.sigmoid(model(features))
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

# Evaluate
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)
        outputs = torch.sigmoid(model(features))
        predictions = (outputs > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
accuracy = correct / total
print("\nEvaluation Accuracy:", accuracy)