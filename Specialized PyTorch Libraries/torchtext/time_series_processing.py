import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# %% [1. Introduction to Time Series Processing]
# Time series processing involves modeling sequential data, e.g., stock prices or sensor readings.
# PyTorch can be used with RNNs, LSTMs, or Transformers for such tasks.

# %% [2. Dummy Dataset]
# Synthetic dataset: 100 sequences, each with 10 timesteps, 2 features, and 1 target value.
torch.manual_seed(42)
num_sequences = 100
seq_length = 10
num_features = 2

# Generate synthetic time series (e.g., sine wave with noise)
t = np.linspace(0, 10, seq_length)
X = np.array([np.sin(t + np.random.rand()) + 0.1 * np.random.randn(seq_length) for _ in range(num_sequences)])
X = np.stack([X, X * 0.5], axis=2)  # 2 features
y = np.mean(X[:, -1, :], axis=1)  # Predict mean of last timestep's features

X = torch.tensor(X, dtype=torch.float32)  # (100, 10, 2)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)  # (100, 1)

class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

dataset = TimeSeriesDataset(X, y)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
print("\nDummy Dataset (first 5 sequences):")
print("Sequence Shape:", X[:5].shape)  # (5, 10, 2)
print("Targets:", y[:5].flatten().tolist())

# %% [3. Preprocessing Time Series Data]
# Normalize the sequences and create sliding windows (simulated).

# Normalize features
mean = X.mean(dim=(0, 1), keepdim=True)
std = X.std(dim=(0, 1), keepdim=True)
X_normalized = (X - mean) / (std + 1e-8)
print("\nNormalized Sequence (first sample, first timestep):", X_normalized[0, 0].tolist())

# Simulate sliding window (already using fixed-length sequences here)
# For real datasets, you'd split long sequences into windows
print("Dataset already in windowed format (10 timesteps)")

# %% [4. Simple RNN for Time Series Prediction]
# Implement an RNN to predict the target value.

class TimeSeriesRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TimeSeriesRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, hidden = self.rnn(x)
        output = self.fc(hidden.squeeze(0))
        return output

# Train RNN
rnn = TimeSeriesRNN(input_size=2, hidden_size=16, output_size=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(rnn.parameters(), lr=0.01)
num_epochs = 5

print("\nTraining RNN:")
for epoch in range(num_epochs):
    rnn.train()
    running_loss = 0.0
    for sequences, targets in dataloader:
        optimizer.zero_grad()
        outputs = rnn(sequences)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * sequences.size(0)
    epoch_loss = running_loss / len(dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# %% [5. LSTM for Time Series]
# Use an LSTM for better handling of long-term dependencies.

class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TimeSeriesLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        output = self.fc(hidden.squeeze(0))
        return output

# Train LSTM
lstm = TimeSeriesLSTM(input_size=2, hidden_size=16, output_size=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(lstm.parameters(), lr=0.01)

print("\nTraining LSTM:")
for epoch in range(num_epochs):
    lstm.train()
    running_loss = 0.0
    for sequences, targets in dataloader:
        optimizer.zero_grad()
        outputs = lstm(sequences)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * sequences.size(0)
    epoch_loss = running_loss / len(dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# %% [6. Transformer for Time Series]
# Implement a Transformer for time series prediction.

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers):
        super(TimeSeriesTransformer, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Pool across timesteps
        x = self.fc(x)
        return x

# Train Transformer
transformer = TimeSeriesTransformer(input_size=2, hidden_size=16, num_heads=2, num_layers=2)
criterion = nn.MSELoss()
optimizer = optim.Adam(transformer.parameters(), lr=0.01)

print("\nTraining Transformer:")
for epoch in range(num_epochs):
    transformer.train()
    running_loss = 0.0
    for sequences, targets in dataloader:
        optimizer.zero_grad()
        outputs = transformer(sequences)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * sequences.size(0)
    epoch_loss = running_loss / len(dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# %% [7. Forecasting with LSTM]
# Use the LSTM to forecast the next value in a sequence.

lstm.eval()
sample_sequence = X_normalized[:1]  # First sequence
with torch.no_grad():
    predicted_value = lstm(sample_sequence)
print("\nLSTM Forecast:")
print("True Target:", y[0].item())
print("Predicted Target:", predicted_value.item())

# Simulate multi-step forecasting (simplified)
forecasts = []
current_sequence = sample_sequence.clone()
for _ in range(3):  # Predict 3 steps ahead
    with torch.no_grad():
        pred = lstm(current_sequence)
    forecasts.append(pred.item())
    # Update sequence (simplified: append prediction as new feature)
    new_timestep = torch.tensor([[pred.item(), pred.item()]], dtype=torch.float32)
    current_sequence = torch.cat([current_sequence[:, 1:, :], new_timestep.unsqueeze(0)], dim=1)
print("Multi-step Forecast (3 steps):", forecasts)

# %% [8. Evaluation]
# Evaluate the LSTM on the dataset.

lstm.eval()
mse_total = 0.0
with torch.no_grad():
    for sequences, targets in dataloader:
        outputs = lstm(sequences)
        mse = ((outputs - targets) ** 2).mean().item()
        mse_total += mse * sequences.size(0)
mse = mse_total / len(dataset)
print("\nLSTM Mean Squared Error:", mse)