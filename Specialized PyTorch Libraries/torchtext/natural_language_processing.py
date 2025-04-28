import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np

# %% [1. Introduction to NLP with PyTorch]
# PyTorch can be used for NLP tasks, often with torchtext or Hugging Face transformers.
# This file focuses on a simple custom dataset and Hugging Face integration.

# Check if transformers is installed
try:
    from transformers import __version__ as transformers_version
    print("transformers version:", transformers_version)
except ImportError:
    print("Hugging Face transformers not installed. Install with: pip install transformers")

# %% [2. Dummy Dataset]
# Synthetic dataset: 100 short sentences with binary sentiment labels (0=negative, 1=positive).
torch.manual_seed(42)
sentences = [
    "I love this movie" if np.random.rand() > 0.5 else "I hate this movie"
    for _ in range(100)
]
labels = [1 if s.startswith("I love") else 0 for s in sentences]

class TextDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_length=32):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            sentence,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = TextDataset(sentences, labels, tokenizer)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

print("\nDummy Dataset (first 5 samples):")
for i in range(5):
    print(f"Sentence: {sentences[i]}, Label: {labels[i]}")

# %% [3. Tokenization and Vocabulary Building]
# Use Hugging Face tokenizer for preprocessing.

# Example tokenization
sample_sentence = sentences[0]
encoded = tokenizer(sample_sentence, return_tensors='pt')
decoded = tokenizer.decode(encoded['input_ids'][0], skip_special_tokens=True)
print("\nTokenization Example:")
print("Original:", sample_sentence)
print("Token IDs:", encoded['input_ids'].squeeze().tolist())
print("Decoded:", decoded)

# Vocabulary size
print("Tokenizer Vocabulary Size:", tokenizer.vocab_size)

# %% [4. Pretrained Language Models (Hugging Face)]
# Use BERT for sequence classification.

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
print("\nBERT Model for Sequence Classification:")
print(model.classifier)

# Test with a single input
sample_input = dataset[0]
output = model(
    input_ids=sample_input['input_ids'].unsqueeze(0),
    attention_mask=sample_input['attention_mask'].unsqueeze(0)
)
print("BERT Output Logits Shape:", output.logits.shape)  # Expected: (1, 2)

# %% [5. Training a Simple RNN for Sentiment Analysis]
# Implement a basic RNN for comparison.

class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        # Apply attention mask
        embedded = embedded * attention_mask.unsqueeze(-1)
        _, hidden = self.rnn(embedded)
        output = self.fc(hidden.squeeze(0))
        return output

# Train RNN
rnn = SimpleRNN(vocab_size=tokenizer.vocab_size, embed_dim=32, hidden_dim=64, output_dim=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn.parameters(), lr=0.001)
num_epochs = 5

print("\nTraining Simple RNN:")
for epoch in range(num_epochs):
    rnn.train()
    running_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = rnn(batch['input_ids'], batch['attention_mask'])
        loss = criterion(outputs, batch['labels'])
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch['input_ids'].size(0)
    epoch_loss = running_loss / len(dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# %% [6. Fine-tuning BERT]
# Fine-tune BERT on the dummy dataset.

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
optimizer = optim.Adam(model.parameters(), lr=2e-5)
num_epochs = 3

print("\nFine-tuning BERT:")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch['input_ids'].size(0)
    epoch_loss = running_loss / len(dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# %% [7. Inference with BERT]
# Perform inference using the fine-tuned BERT model.

model.eval()
sample_sentence = "I love this movie"
encoded = tokenizer(sample_sentence, return_tensors='pt', padding=True, truncation=True, max_length=32)
with torch.no_grad():
    outputs = model(**encoded)
    probs = torch.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probs, dim=-1).item()
print("\nInference with BERT:")
print("Sentence:", sample_sentence)
print("Predicted Class (0=Negative, 1=Positive):", predicted_class)
print("Probabilities:", probs[0].tolist())

# %% [8. Evaluation]
# Evaluate the fine-tuned BERT model.

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in dataloader:
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        _, predicted = torch.max(outputs.logits, 1)
        total += batch['labels'].size(0)
        correct += (predicted == batch['labels']).sum().item()
accuracy = correct / total
print("\nBERT Test Accuracy:", accuracy)