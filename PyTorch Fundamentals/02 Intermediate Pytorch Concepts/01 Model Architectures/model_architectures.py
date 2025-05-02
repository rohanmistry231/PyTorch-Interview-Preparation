import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

# %% [1. Feedforward Neural Networks (FNNs)]
# FNNs are fully connected networks for tasks like regression or classification.

class FNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(FNN, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Instantiate and test FNN
fnn = FNN(input_size=2, hidden_sizes=[8, 4], output_size=1)
sample_input = torch.randn(5, 2)
output = fnn(sample_input)
print("FNN Output Shape:", output.shape)  # Expected: (5, 1)

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

# %% [3. Convolutional Neural Networks (CNNs)]
# CNNs are used for image-related tasks like classification.

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, num_classes)  # For 28x28 input
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# Test CNN with dummy image data
cnn = SimpleCNN(num_classes=2)
dummy_images = torch.randn(4, 1, 28, 28)  # Batch of 4 grayscale 28x28 images
output = cnn(dummy_images)
print("\nCNN Output Shape:", output.shape)  # Expected: (4, 2)

# %% [4. Recurrent Neural Networks (RNNs)]
# RNNs are used for sequence data, e.g., time series or text.

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, hidden = self.rnn(x)  # Last hidden state
        output = self.fc(hidden.squeeze(0))
        return output

# Test RNN with sequence data
rnn = SimpleRNN(input_size=3, hidden_size=10, output_size=1)
sequence_data = torch.randn(5, 10, 3)  # 5 sequences, 10 timesteps, 3 features
output = rnn(sequence_data)
print("\nRNN Output Shape:", output.shape)  # Expected: (5, 1)

# %% [5. Transfer Learning]
# Use pretrained models for tasks like image classification.

# Load pretrained ResNet18
resnet = models.resnet18(pretrained=True)
# Modify the final layer for binary classification
resnet.fc = nn.Linear(resnet.fc.in_features, 2)
print("\nModified ResNet18 Final Layer:", resnet.fc)

# Test with dummy image batch
dummy_images = torch.randn(4, 3, 224, 224)  # RGB 224x224 images
output = resnet(dummy_images)
print("ResNet Output Shape:", output.shape)  # Expected: (4, 2)

# Freeze layers for feature extraction
for param in resnet.parameters():
    param.requires_grad = False
resnet.fc.weight.requires_grad = True  # Train only the final layer
print("\nTrainable Parameters in Frozen ResNet:")
for name, param in resnet.named_parameters():
    if param.requires_grad:
        print(name)

# %% [6. Attention Mechanisms]
# Implement a simple self-attention layer.

class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.scale = embed_size ** 0.5
    
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

# Test self-attention
attention = SelfAttention(embed_size=16)
dummy_sequence = torch.randn(5, 10, 16)  # 5 sequences, 10 tokens, 16 dims
output, weights = attention(dummy_sequence)
print("\nSelf-Attention Output Shape:", output.shape)  # Expected: (5, 10, 16)
print("Attention Weights Shape:", weights.shape)  # Expected: (5, 10, 10)

# %% [7. Training a CNN on Dummy Dataset]
# Train a simplified CNN on the dummy dataset (reshaped as images).

class BinaryCNN(nn.Module):
    def __init__(self):
        super(BinaryCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc = nn.Linear(4 * 1 * 1, 1)  # Adjusted for 4x4 input
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Reshape dummy dataset as 4x4 grayscale images
X_images = X.reshape(-1, 1, 4, 4)  # (100, 1, 4, 4)
dataset_images = CustomDataset(X_images, y)
dataloader_images = DataLoader(dataset_images, batch_size=16, shuffle=True)

# Train the CNN
cnn = BinaryCNN()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.01)
num_epochs = 5

print("\nTraining CNN:")
for epoch in range(num_epochs):
    cnn.train()
    running_loss = 0.0
    for features, labels in dataloader_images:
        optimizer.zero_grad()
        outputs = cnn(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * features.size(0)
    epoch_loss = running_loss / len(dataset_images)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# %% [8. Evaluation]
# Evaluate the trained CNN.

cnn.eval()
correct = 0
total = 0
with torch.no_grad():
    for features, labels in dataloader_images:
        outputs = torch.sigmoid(cnn(features))
        predictions = (outputs > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
accuracy = correct / total
print("\nCNN Training Accuracy:", accuracy)