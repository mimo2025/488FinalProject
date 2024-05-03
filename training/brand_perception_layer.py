from transformers import AutoTokenizer
from modules.BrandPerceptionModel import BrandPerceptionModel
from brand_perception_data_module import BrandPerceptionDataModule
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from pytorch_lightning import Trainer

print("Starting")

# Load data
print("Loading data")
df = pd.read_csv('Labeled_Df.csv')
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

# Initialize data module
print("Initializing data module")
brand_perception_dm = BrandPerceptionDataModule(train_df, val_df)

# Configurations
config = {
    'model_name': 'SamLowe/roberta-base-go_emotions',
    'n_labels_bp': 6,
    'lr': 1.5e-6,
    'weight_decay': 0.001,
    'warmup': 0.2,
    'train_size': len(train_df),
    'batch_size': 128,
    'n_epochs': 100
}

# Initialize model
print("Initizlaing model")
model = BrandPerceptionModel(config)
# Initialize the trainer with GPU configuration
print("Initizlaing trainer")
trainer = Trainer(
    max_epochs=config['n_epochs'],
    num_sanity_val_steps=50,
    accelerator='gpu',  # Automatically choose the best backend (GPU/CPU)
    strategy='ddp'  # Use data parallelism
)

# Start training
print("Starting trainer")
trainer.fit(model, brand_perception_dm)