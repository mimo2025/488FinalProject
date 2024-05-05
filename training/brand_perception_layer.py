import sys
import os
import torch
from torch.utils.data import DataLoader
import pickle
import pytorch_lightning as pl
from modules.BrandPerceptionModel import BrandPerceptionModel

# Clear any previous GPU memory
torch.cuda.empty_cache()

# Add the parent directory to the system path
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_directory)

print("Starting")

# Load Dataset
print("Loading datasets")
with open('/nas/longleaf/home/aryonna/488FinalProject/datasetss/train_dataset.pkl', 'rb') as f:
    train_dataset = pickle.load(f)

with open('/nas/longleaf/home/aryonna/488FinalProject/datasetss/val_dataset.pkl', 'rb') as f:
    val_dataset = pickle.load(f)

print('Creating dataloaders')
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Configurations
config = {
    'model_name': 'SamLowe/roberta-base-go_emotions',
    'n_labels_bp': 6,
    'batch_size': 8,
    'lr': 1.5e-5,
    'warmup': 0.2, 
    'train_size': len(train_loader),
    'weight_decay': 0.001,
    'n_epochs': 10
}

print('Loading model from checkpoint')
model = BrandPerceptionModel.load_from_checkpoint(
    "/nas/longleaf/home/aryonna/488FinalProject/models/brand_perception_model_checkpoint.ckpt",
    config=config
)

accumulation_steps = 2

trainer = pl.Trainer(
    max_epochs=config['n_epochs'],
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    precision=16,  # Mixed precision
    accumulate_grad_batches=accumulation_steps
)

print("Beginning training")
trainer.fit(model, train_loader, val_loader)

# Save new model
print("Saving model")
trainer.save_checkpoint("/nas/longleaf/home/aryonna/488FinalProject/models/brand_perception_model_checkpoint.ckpt")
