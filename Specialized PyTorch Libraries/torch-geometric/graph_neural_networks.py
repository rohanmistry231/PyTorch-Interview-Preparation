import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
import numpy as np

# %% [1. Introduction to Graph Neural Networks]
# Graph Neural Networks (GNNs) process graph-structured data using libraries like PyTorch Geometric.
# They are used for tasks like node classification, link prediction, and graph classification.

# Check torch_geometric version
print("torch_geometric version:", torch_geometric.__version__)

# %% [2. Dummy Dataset]
# Synthetic dataset: 100 small graphs, each with 10 nodes, 2 features, and a binary label.
torch.manual_seed(42)
num_graphs = 100
num_nodes = 10
num_features = 2

graphs = []
for _ in range(num_graphs):
    # Node features: random 2D features
    x = torch.randn(num_nodes, num_features)
    # Edge index: fully connected graph (simplified)
    edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
    # Graph label: binary based on sum of node features
    y = torch.tensor([1 if x.sum() > 0 else 0], dtype=torch.long)
    # Create graph
    graph = Data(x=x, edge_index=edge_index, y=y)
    graphs.append(graph)

class GraphDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx]

dataset = GraphDataset(graphs)
dataloader = GeoDataLoader(dataset, batch_size=16, shuffle=True)
print("\nDummy Dataset (first graph):")
print("Node Features Shape:", graphs[0].x.shape)  # (10, 2)
print("Edge Index Shape:", graphs[0].edge_index.shape)  # (2, 45)
print("Label:", graphs[0].y.item())

# %% [3. torch_geometric Datasets]
# Load a real graph dataset (e.g., Cora for node classification).

from torch_geometric.datasets import Planetoid
cora = Planetoid(root='./data', name='Cora')
print("\nCora Dataset:")
print("Number of graphs:", len(cora))
print("Number of nodes:", cora[0].num_nodes)
print("Number of edges:", cora[0].num_edges)
print("Node Features Shape:", cora[0].x.shape)
print("Labels Shape:", cora[0].y.shape)

# %% [4. Graph Convolutional Network (GCN)]
# Implement a GCN for graph classification.

from torch_geometric.nn import GCNConv, global_mean_pool

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = global_mean_pool(x, batch)  # Pool node features to graph-level
        x = self.fc(x)
        return x

# Test GCN
gcn = GCN(input_dim=2, hidden_dim=16, output_dim=2)
sample_batch = next(iter(dataloader))
output = gcn(sample_batch)
print("\nGCN Output Shape:", output.shape)  # Expected: (16, 2)

# %% [5. Graph Attention Network (GAT)]
# Implement a GAT for graph classification.

from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x

# Test GAT
gat = GAT(input_dim=2, hidden_dim=16, output_dim=2)
output = gat(sample_batch)
print("\nGAT Output Shape:", output.shape)  # Expected: (16, 2)

# %% [6. Training GCN on Dummy Dataset]
# Train the GCN for graph classification.

gcn = GCN(input_dim=2, hidden_dim=16, output_dim=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(gcn.parameters(), lr=0.01)
num_epochs = 5

print("\nTraining GCN:")
for epoch in range(num_epochs):
    gcn.train()
    running_loss = 0.0
    for data in dataloader:
        optimizer.zero_grad()
        outputs = gcn(data)
        loss = criterion(outputs, data.y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.num_graphs
    epoch_loss = running_loss / len(dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# %% [7. Node Classification with GCN on Cora]
# Train a GCN for node classification on the Cora dataset.

class NodeGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NodeGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Train on Cora
cora_data = cora[0]
node_gcn = NodeGCN(input_dim=cora_data.num_features, hidden_dim=16, output_dim=cora.num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(node_gcn.parameters(), lr=0.01)

print("\nTraining Node GCN on Cora:")
for epoch in range(5):  # Short training for demo
    node_gcn.train()
    optimizer.zero_grad()
    outputs = node_gcn(cora_data)
    loss = criterion(outputs[cora_data.train_mask], cora_data.y[cora_data.train_mask])
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/5, Loss: {loss.item():.4f}")

# %% [8. Evaluation]
# Evaluate the GCN on the dummy dataset.

gcn.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in dataloader:
        outputs = gcn(data)
        _, predicted = torch.max(outputs, 1)
        total += data.y.size(0)
        correct += (predicted == data.y).sum().item()
accuracy = correct / total
print("\nGCN Accuracy on Dummy Dataset:", accuracy)

# Evaluate Node GCN on Cora
node_gcn.eval()
with torch.no_grad():
    outputs = node_gcn(cora_data)
    _, predicted = torch.max(outputs[cora_data.test_mask], 1)
    correct = (predicted == cora_data.y[cora_data.test_mask]).sum().item()
    total = cora_data.test_mask.sum().item()
accuracy = correct / total
print("Node GCN Accuracy on Cora Test Set:", accuracy)