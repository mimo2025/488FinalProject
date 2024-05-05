import torch
from transformers import AutoModel, AdamW, get_cosine_schedule_with_warmup
import torch.nn as nn
import math
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics.classification import MultilabelAccuracy, MultilabelF1Score


class BrandPerceptionModel(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        # Load the pretrained model (e.g., a RoBERTa model trained on GoEmotions)
        self.config = config
        self.pretrained_model = AutoModel.from_pretrained(config['model_name'])

        # Classifier for emotions (outputting 28 classes)
        self.emotion_classifier = nn.Linear(self.pretrained_model.config.hidden_size, 28)

        # Classifier for brand perception aspects
        self.brand_perception_classifier = nn.Linear(self.pretrained_model.config.hidden_size, config['n_labels_bp'])
        self.num_emotion_labels = 28
        self.num_brand_labels = 6
        self.accuracy_emotion = MultilabelAccuracy(num_labels=self.num_emotion_labels)
        self.f1_score_emotion = MultilabelF1Score(num_labels=self.num_emotion_labels)
        self.accuracy_brand = MultilabelAccuracy(num_labels=self.num_brand_labels)
        self.f1_score_brand = MultilabelF1Score(num_labels=self.num_brand_labels)
        self.loss_func = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, input_ids, attention_mask, labels_emotion=None, labels_brand=None):
        outputs = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden_state = outputs.last_hidden_state
        mean_pooled = last_hidden_state.mean(dim=1)
        emotion_logits = self.emotion_classifier(mean_pooled)
        brand_perception_logits = self.brand_perception_classifier(mean_pooled)

        loss = 0
        
        print("Emotion Logits Size:", emotion_logits.size())
        print("Brand Logits Size:", brand_perception_logits.size())

        if labels_emotion is not None:
            print("Labels Emotion Size:", labels_emotion.size())
            emotion_loss = F.binary_cross_entropy_with_logits(emotion_logits, labels_emotion)
            loss += emotion_loss

        if labels_brand is not None:
            print("Labels Brand Size:", labels_brand.size())
            brand_loss = F.binary_cross_entropy_with_logits(brand_perception_logits, labels_brand)
            loss += brand_loss

        return loss, emotion_logits, brand_perception_logits

    
    def training_step(self, batch, batch_idx):
        loss, emotion_logits, brand_logits = self(
            input_ids=batch['input_ids'], 
            attention_mask=batch['attention_mask'], 
            labels_emotion=batch['labels_emotion'],
            labels_brand=batch['labels_brand']
        )
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "emotion_logits": emotion_logits, "brand_logits": brand_logits}

    def validation_step(self, batch, batch_idx):
        loss, emotion_logits, brand_logits = self(
            input_ids=batch['input_ids'], 
            attention_mask=batch['attention_mask'], 
            labels_emotion=batch['labels_emotion'],
            labels_brand=batch['labels_brand']
        )
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "emotion_logits": emotion_logits, "brand_logits": brand_logits}


    def predict_step(self, batch):  
        _, emotion_logits, brand_logits = self(
            input_ids=batch['input_ids'], 
            attention_mask=batch['attention_mask'], 
            labels_emotion=batch['labels_emotion'],
            labels_brand=batch['labels_brand']
        )
        
        # Apply sigmoid to the logits to convert to probabilities
        emotion_probs = torch.sigmoid(emotion_logits)
        brand_probs = torch.sigmoid(brand_logits)

        return {"emotion_probs": emotion_probs, "brand_probs": brand_probs}
    
    def test_step(self, batch, batch_idx):
        loss, emotion_logits, brand_perception_logits = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels_emotion=batch['labels_emotion'],
            labels_brand=batch['labels_brand']
        )
        
        emotion_preds = torch.sigmoid(emotion_logits)
        brand_preds = torch.sigmoid(brand_perception_logits)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_accuracy_emotion", self.accuracy_emotion(emotion_preds, batch['labels_emotion']), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_f1_score_emotion", self.f1_score_emotion(emotion_preds, batch['labels_emotion']), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_accuracy_brand", self.accuracy_brand(brand_preds, batch['labels_brand']), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_f1_score_brand", self.f1_score_brand(brand_preds, batch['labels_brand']), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "emotion_logits": emotion_logits, "brand_perception_logits": brand_perception_logits}

    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        total_steps = self.config['train_size'] // self.config['batch_size'] * self.config['n_epochs']
        warmup_steps = int(total_steps * self.config['warmup'])
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
