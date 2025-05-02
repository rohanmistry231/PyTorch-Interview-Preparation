import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# %% [1. Introduction to Advanced Model Architectures]
# Advanced architectures like Transformers, VAEs, and GANs handle complex tasks.
# This file demonstrates simplified versions of these models.

# %% [2. Dummy Dataset]
# Synthetic dataset: 100 samples, 2 features for simple tasks, and image-like data for others.
torch.manual_seed(42)
X = torch.randn(100, 2)  # For Transformer, VAE
y = (X[:, 0] + X[:, 1] > 0).float().reshape(-1, 1)  # Binary labels
X_images = torch.randn(100, 1, 28, 28)  # For GAN (28x28 grayscale images)

class CustomDataset(Dataset):
    def __init__(self, features, labels=None):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]

dataset = CustomDataset(X, y)
image_dataset = CustomDataset(X_images)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
image_dataloader = DataLoader(image_dataset, batch_size=16, shuffle=True)
print("\nDummy Dataset (first 5 samples):")
print("Features (X):\n", X[:5])
print("Image Dataset Shape (first sample):", X_images[0].shape)

# %% [3. Transformers: Simple Transformer Encoder]
# Implement a basic Transformer encoder for sequence classification.

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = self.embedding(x)  # (batch, seq, hidden_dim)
        x = x.permute(1, 0, 2)  # (seq, batch, hidden_dim)
        x = self.transformer(x)
        x = x.mean(dim=0)  # Pool across sequence
        x = self.fc(x)
        return x

# Test Transformer
transformer = TransformerEncoder(input_dim=2, hidden_dim=16, num_heads=2, num_layers=2)
sample_input = torch.randn(5, 10, 2)  # 5 sequences, 10 timesteps, 2 features
output = transformer(sample_input)
print("\nTransformer Output Shape:", output.shape)  # Expected: (5, 1)

# %% [4. Generative Models: Variational Autoencoder (VAE)]
# Implement a VAE for learning latent representations.

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # Mean and log-variance
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

# Test VAE
vae = VAE(input_dim=2, hidden_dim=8, latent_dim=4)
sample_input = torch.randn(5, 2)
recon, mu, logvar = vae(sample_input)
print("\nVAE Reconstruction Shape:", recon.shape)  # Expected: (5, 2)

# %% [5. Generative Models: Generative Adversarial Network (GAN)]
# Implement a simple GAN for generating images.

class Generator(nn.Module):
    def __init__(self, latent_dim, feature_maps, image_channels):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature_maps * 4, 4, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(feature_maps, image_channels, 4, 2, 1),
            nn.Tanh()
        )
    
    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, image_channels, feature_maps):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(image_channels, feature_maps, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(feature_maps * 4, 1, 4, 1, 0),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x).view(-1, 1)

# Test GAN
generator = Generator(latent_dim=100, feature_maps=64, image_channels=1)
discriminator = Discriminator(image_channels=1, feature_maps=64)
z = torch.randn(5, 100)
fake_images = generator(z)
disc_output = discriminator(fake_images)
print("\nGAN Generator Output Shape:", fake_images.shape)  # Expected: (5, 1, 28, 28)
print("Discriminator Output Shape:", disc_output.shape)  # Expected: (5, 1)

# %% [6. Training a Transformer]
# Train the Transformer on the dummy dataset.

transformer = TransformerEncoder(input_dim=2, hidden_dim=16, num_heads=2, num_layers=2)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(transformer.parameters(), lr=0.01)
num_epochs = 5

print("\nTraining Transformer:")
for epoch in range(num_epochs):
    transformer.train()
    running_loss = 0.0
    for features, labels in dataloader:
        optimizer.zero_grad()
        # Reshape for sequence (batch, seq=1, features)
        features = features.unsqueeze(1)  # (batch, 1, 2)
        outputs = transformer(features).squeeze(-1)
        loss = criterion(outputs, labels.squeeze(-1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * features.size(0)
    epoch_loss = running_loss / len(dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# %% [7. Training a GAN]
# Train the GAN on the image dataset.

generator = Generator(latent_dim=100, feature_maps=64, image_channels=1)
discriminator = Discriminator(image_channels=1, feature_maps=64)
criterion = nn.BCELoss()
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
num_epochs = 5

print("\nTraining GAN:")
for epoch in range(num_epochs):
    generator.train()
    discriminator.train()
    g_loss_total = 0.0
    d_loss_total = 0.0
    for images in image_dataloader:
        batch_size = images.size(0)
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
        
        # Train Discriminator
        d_optimizer.zero_grad()
        real_output = discriminator(images)
        d_real_loss = criterion(real_output, real_labels)
        z = torch.randn(batch_size, 100)
        fake_images = generator(z)
        fake_output = discriminator(fake_images.detach())
        d_fake_loss = criterion(fake_output, fake_labels)
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        d_optimizer.step()
        
        # Train Generator
        g_optimizer.zero_grad()
        fake_output = discriminator(fake_images)
        g_loss = criterion(fake_output, real_labels)
        g_loss.backward()
        g_optimizer.step()
        
        g_loss_total += g_loss.item() * batch_size
        d_loss_total += d_loss.item() * batch_size
    
    g_epoch_loss = g_loss_total / len(image_dataset)
    d_epoch_loss = d_loss_total / len(image_dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, G Loss: {g_epoch_loss:.4f}, D Loss: {d_epoch_loss:.4f}")

# %% [8. Evaluation]
# Evaluate the Transformer on the dummy dataset.

transformer.eval()
correct = 0
total = 0
with torch.no_grad():
    for features, labels in dataloader:
        features = features.unsqueeze(1)
        outputs = torch.sigmoid(transformer(features)).squeeze(-1)
        predictions = (outputs > 0.5).float()
        correct += (predictions == labels.squeeze(-1)).sum().item()
        total += labels.size(0)
accuracy = correct / total
print("\nTransformer Accuracy:", accuracy)