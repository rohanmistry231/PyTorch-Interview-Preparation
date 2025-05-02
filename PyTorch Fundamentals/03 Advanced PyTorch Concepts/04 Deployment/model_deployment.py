import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.jit
import torch.onnx
import numpy as np

# %% [1. Introduction to Model Deployment]
# Deployment involves exporting models for production, serving them via APIs,
# and optimizing for edge devices.

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
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
print("\nDummy Dataset (first 5 samples):")
print("Features (X):\n", X[:5])
print("Labels (y):\n", y[:5])

# %% [3. Training the Model]
# Train a model to deploy later.

model = SimpleNet(input_size=2, hidden_size=8, output_size=1)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 5

print("\nTraining Model:")
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

# %% [4. Exporting to ONNX]
# Export the model to ONNX format for cross-platform compatibility.

model.eval()
dummy_input = torch.randn(1, 2)  # Example input for tracing
onnx_path = "model.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)
print("\nModel exported to ONNX:", onnx_path)

# Verify ONNX model (simulated check)
import onnx
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print("ONNX model check passed")

# %% [5. Exporting to TorchScript]
# Export the model to TorchScript for optimized inference.

# Tracing
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save("model_traced.pt")
print("\nModel exported to TorchScript (traced): model_traced.pt")

# Scripting
scripted_model = torch.jit.script(model)
scripted_model.save("model_scripted.pt")
print("Model exported to TorchScript (scripted): model_scripted.pt")

# Test TorchScript model
loaded_traced_model = torch.jit.load("model_traced.pt")
output = torch.sigmoid(loaded_traced_model(dummy_input))
print("TorchScript Model Output:", output.detach().numpy())

# %% [6. Serving Models (Simulated Flask API)]
# Simulate a Flask API for model inference.

from flask import Flask, request, jsonify
import threading

app = Flask(__name__)
model = loaded_traced_model
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_data = torch.tensor(data['input'], dtype=torch.float32)
    with torch.no_grad():
        output = torch.sigmoid(model(input_data))
        prediction = (output > 0.5).float().numpy().tolist()
    return jsonify({'prediction': prediction})

# Simulate running the Flask server in a separate thread
def run_flask():
    app.run(debug=False, use_reloader=False, port=5000)

# Start Flask server (commented out for actual execution, simulated here)
# threading.Thread(target=run_flask, daemon=True).start()
print("\nSimulated Flask API endpoint: /predict")
# Simulate a POST request
sample_input = {'input': [[1.0, 2.0]]}
print("Simulated POST input:", sample_input)
print("Simulated Response:", {'prediction': [1.0]})  # Mocked output

# %% [7. Deploying on Edge Devices]
# Optimize model for edge devices using quantization.

model.eval()
model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
quantized_model = torch.quantization.prepare(model, inplace=False)
quantized_model = torch.quantization.convert(quantized_model, inplace=False)
print("\nQuantized Model for Edge Deployment:")
print(quantized_model)

# Test quantized model
output = torch.sigmoid(quantized_model(dummy_input))
print("Quantized Model Output Shape:", output.shape)

# Save quantized model for edge deployment
torch.jit.save(torch.jit.script(quantized_model), "model_quantized.pt")
print("Quantized Model Saved: model_quantized.pt")

# %% [8. Production Optimization: Model Compression]
# Apply pruning to reduce model size.

import torch.nn.utils.prune as prune

# Prune 50% of weights in layer1
prune.random_unstructured(model.layer1, name="weight", amount=0.5)
prune.remove(model.layer1, "weight")  # Make pruning permanent
print("\nAfter Pruning (Layer1 Weights):")
print("Non-zero Weights:", torch.count_nonzero(model.layer1.weight).item())

# Train one epoch to fine-tune after pruning
optimizer = optim.Adam(model.parameters(), lr=0.01)
model.train()
for features, labels in dataloader:
    optimizer.zero_grad()
    outputs = torch.sigmoid(model(features))
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    print("Fine-tuning Loss after Pruning:", loss.item())
    break

# Evaluate pruned model
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
print("\nPruned Model Accuracy:", accuracy)