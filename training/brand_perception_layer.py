import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from modules import BrandPerceptionModel
from datasets.brand_perception_dataset import BrandPerceptionDataset

# Define your training parameters
num_epochs = 5
batch_size = 32
learning_rate = 2e-5

# Example data (replace with your actual data)
texts = [...]  # List of text inputs from scrapped data
aspect_labels = [...]  # List of 1's and 0's correspoding to if aspect of brand perception was found

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")

# Define maximum sequence length
max_length = 128

# Create dataset
train_dataset = BrandPerceptionDataset(texts, aspect_labels, tokenizer, max_length)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define your loss function for aspect identification
criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss for multi-label classification

model = BrandPerceptionModel()

# Define your optimizer
optimizer = AdamW(model.aspect_classifier.parameters(), lr=learning_rate)  # Only optimize parameters of aspect identification layer

# Define your learning rate scheduler
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Set model to training mode
model.train()

# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    
    for batch in train_dataloader:
        # Extract input data and labels from batch
        input_ids, attention_mask, aspect_labels = batch
        
        # Forward pass
        aspect_logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Compute loss
        loss = criterion(aspect_logits, aspect_labels.float())
        total_loss += loss.item()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.aspect_classifier.parameters(), 1.0)  # Gradient clipping to prevent exploding gradients
        optimizer.step()
        scheduler.step()

    # Calculate average loss for the epoch
    avg_loss = total_loss / len(train_dataloader)
    
    # Print epoch loss
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
