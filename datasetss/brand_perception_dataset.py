from torch.utils.data import Dataset
import torch
from transformers import RobertaTokenizer


class BrandPerceptionDataset(Dataset):
    def __init__(self, texts, emotion_labels, brand_labels):
        self.texts = texts
        self.emotion_labels = torch.tensor(emotion_labels, dtype=torch.float32)
        self.brand_labels = torch.tensor(brand_labels, dtype=torch.float32)
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        emotion_label = self.emotion_labels[idx]
        brand_label = self.brand_labels[idx]
        
        # Tokenize the text
        inputs = self.tokenizer(text, return_tensors='pt', 
                                padding='max_length', truncation=True, 
                                max_length=512)
        
        # Extract input_ids and attention_mask
        input_ids = inputs['input_ids'].squeeze()  # Remove batch dimension
        attention_mask = inputs['attention_mask'].squeeze()  # Remove batch dimension

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels_emotion': emotion_label,
            'labels_brand': brand_label
        }

