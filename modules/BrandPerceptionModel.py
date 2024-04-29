import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

class BrandPerceptionModel(nn.Module):
    def __init__(self, num_aspects=7, num_emotions=28):
        super(BrandPerceptionModel, self).__init__()
        # Load the pretrained emotion classification model
        self.emotion_model = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")
        
        # Aspect identification layer
        self.aspect_classifier = nn.Linear(self.emotion_model.config.hidden_size, num_aspects)

    def forward(self, input_ids, attention_mask):
        # Emotion classification
        emotion_outputs = self.emotion_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = emotion_outputs.pooler_output
        
        # Aspect identification
        aspect_logits = self.aspect_classifier(pooled_output)
        
        return aspect_logits, emotion_outputs.logits