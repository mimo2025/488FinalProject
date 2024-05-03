import torch
from torch.utils.data import Dataset

class BrandPerceptionDataset(Dataset):
    def __init__(self, texts, emotional_scores, brand_labels):
        """
        Initialize dataset.
        Args:
            texts (list of str): The text samples.
            emotional_scores (list of lists): The emotional scores from the emotion detection model.
            brand_labels (list of lists): The binary labels for brand dimensions.
        """
        self.texts = texts
        self.emotional_scores = emotional_scores
        self.brand_labels = brand_labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        emotional_score = torch.tensor(self.emotional_scores[idx], dtype=torch.float32)
        brand_label = torch.tensor(self.brand_labels[idx], dtype=torch.int64)
        
        return {
            'text': text,
            'emotional_score': emotional_score,
            'brand_label': brand_label
        }

