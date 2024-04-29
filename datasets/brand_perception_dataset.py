from torch.utils.data import Dataset

class BrandPerceptionDataset(Dataset):
    def __init__(self, texts, aspect_labels, tokenizer, max_length):
        self.texts = texts
        self.aspect_labels = aspect_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        aspect_label = self.aspect_labels[idx]
        
        # Tokenize text and convert to input_ids and attention_mask
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        return input_ids, attention_mask, aspect_label
