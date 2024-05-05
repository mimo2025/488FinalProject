from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer
from datasetss.brand_perception_dataset import BrandPerceptionDataset

class BrandPerceptionDataModule(pl.LightningDataModule):
  def __init__(self, train_df, val_df, batch_size: int = 16, max_token_length: int = 128,  model_name='roberta-base'):
    super().__init__()
    self.train_df = train_df
    self.val_df = val_df
    self.batch_size = batch_size
    self.max_token_length = max_token_length
    self.model_name = model_name
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)

  def setup(self, stage = None):
    if stage in (None, "fit"):
      self.train_dataset = BrandPerceptionDataset(self.train_df, tokenizer=self.tokenizer)
      self.val_dataset = BrandPerceptionDataset(self.val_df, tokenizer=self.tokenizer)
    if stage == 'predict':
        self.val_dataset = BrandPerceptionDataset(self.val_df, tokenizer=self.tokenizer)

  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size = self.batch_size, num_workers=4, shuffle=True, collate_fn=None)

  def val_dataloader(self):
    return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers=4, shuffle=False, collate_fn=None)

  def predict_dataloader(self):
    return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers=4, shuffle=False, collate_fn=None)