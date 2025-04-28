import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as transforms
import numpy as np

# %% [1. Introduction to torchaudio]
# torchaudio provides tools for audio processing, including datasets, transforms, and models.
# It integrates seamlessly with PyTorch for tasks like speech recognition.

# Check torchaudio version
print("torchaudio version:", torchaudio.__version__)

# %% [2. Dummy Dataset]
# Synthetic dataset: 100 audio-like waveforms (1-second, 16kHz) with binary labels.
torch.manual_seed(42)
sample_rate = 16000
duration = 1.0
num_samples = int(sample_rate * duration)
waveforms = torch.randn(100, 1, num_samples)  # 100 samples, 1 channel, 16k samples
labels = (torch.randn(100) > 0).long()  # Binary labels (0 or 1)

class AudioDataset(Dataset):
    def __init__(self, waveforms, labels, transform=None):
        self.waveforms = waveforms
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.waveforms)
    
    def __getitem__(self, idx):
        waveform = self.waveforms[idx]
        label = self.labels[idx]
        if self.transform:
            waveform = self.transform(waveform)
        return waveform, label

dataset = AudioDataset(waveforms, labels)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
print("\nDummy Dataset (first 5 samples):")
print("Waveform Shape:", waveforms[:5].shape)  # (5, 1, 16000)
print("Labels:", labels[:5].tolist())

# %% [3. torchaudio Datasets]
# Load a real audio dataset (e.g., SpeechCommands).

speech_commands = torchaudio.datasets.SPEECHCOMMANDS(
    root='./data',
    download=True,
    subset='training'
)
print("\nSpeechCommands Dataset:")
print("Number of samples:", len(speech_commands))
waveform, sample_rate, label, *_ = speech_commands[0]
print("Sample Waveform Shape:", waveform.shape)
print("Sample Rate:", sample_rate)
print("Sample Label:", label)

# Create a DataLoader for SpeechCommands
speech_loader = DataLoader(speech_commands, batch_size=16, shuffle=True, num_workers=2)

# %% [4. Audio Transforms]
# Apply torchaudio transforms for preprocessing (e.g., spectrograms).

# Transform to compute Mel spectrogram
mel_transform = transforms.MelSpectrogram(
    sample_rate=16000,
    n_mels=64,
    n_fft=400,
    hop_length=160
)

# Apply transform to dummy dataset
transformed_dataset = AudioDataset(waveforms, labels, transform=mel_transform)
print("\nMel Spectrogram Transform:")
waveform, label = transformed_dataset[0]
print("Transformed Waveform Shape:", waveform.shape)  # Expected: (1, 64, 101)

# Example with resampling
resampler = transforms.Resample(orig_freq=16000, new_freq=8000)
resampled_waveform = resampler(waveforms[0])
print("Resampled Waveform Shape (8kHz):", resampled_waveform.shape)  # Expected: (1, 8000)

# %% [5. Audio Processing Utilities]
# Use torchaudio for audio I/O and feature extraction.

# Save a waveform as an audio file
torchaudio.save("sample_audio.wav", waveforms[0], sample_rate=16000)
print("\nSaved sample audio to: sample_audio.wav")

# Load and verify
loaded_waveform, loaded_sample_rate = torchaudio.load("sample_audio.wav")
print("Loaded Waveform Shape:", loaded_waveform.shape)
print("Loaded Sample Rate:", loaded_sample_rate)

# Compute MFCC features
mfcc_transform = transforms.MFCC(sample_rate=16000, n_mfcc=13)
mfcc_features = mfcc_transform(waveforms[0])
print("MFCC Features Shape:", mfcc_features.shape)

# %% [6. Simple Audio Classification Model]
# Define and train a CNN for audio classification on Mel spectrograms.

class AudioCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 16 * 25, num_classes)  # Adjusted for Mel spectrogram
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Train the model
model = AudioCNN(num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 5

print("\nTraining Audio CNN on Dummy Dataset:")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for waveforms, labels in dataloader:
        mel_specs = mel_transform(waveforms)  # Convert to Mel spectrograms
        optimizer.zero_grad()
        outputs = model(mel_specs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * waveforms.size(0)
    epoch_loss = running_loss / len(dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# %% [7. Transfer Learning with Pretrained Models]
# Use a pretrained audio model (simulated with a custom model).

# Simulate a pretrained model by modifying a CNN
pretrained_model = AudioCNN(num_classes=10)  # Pretrained on 10 classes
pretrained_model.fc = nn.Linear(pretrained_model.fc.in_features, 2)  # Adapt for binary classification

# Fine-tune only the final layer
for param in pretrained_model.parameters():
    param.requires_grad = False
pretrained_model.fc.weight.requires_grad = True
pretrained_model.fc.bias.requires_grad = True

optimizer = optim.Adam(pretrained_model.fc.parameters(), lr=0.001)
print("\nFine-tuning Pretrained Audio CNN:")
for epoch in range(3):
    pretrained_model.train()
    running_loss = 0.0
    for waveforms, labels in dataloader:
        mel_specs = mel_transform(waveforms)
        optimizer.zero_grad()
        outputs = pretrained_model(mel_specs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * waveforms.size(0)
    epoch_loss = running_loss / len(dataset)
    print(f"Epoch {epoch+1}/3, Loss: {epoch_loss:.4f}")

# %% [8. Evaluation]
# Evaluate the trained CNN on the dummy dataset.

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for waveforms, labels in dataloader:
        mel_specs = mel_transform(waveforms)
        outputs = model(mel_specs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = correct / total
print("\nAudio CNN Accuracy:", accuracy)