# Brand Perception Sentiment Analysis Model
This project focuses on building a synthetic expert using a fine-tuned RoBERTa model for sentiment analysis and brand perception classification for luxury fashion brands based on TikTok data. The model is designed to output insights into brand perception across multiple dimensions and classify emotions based on the GoEmotions dataset.

## Project Overview
### Key Features
- Multi-class Brand Perception Classifier: Classifies TikTok comments and captions about luxury fashion brands into aspects such as product quality, reputation & heritage, customer service, social impact, ethical practices, and sustainability.
- Emotion Classification: Utilizes the GoEmotions simplified dataset to identify emotions within the text.

## Model Overview
- The model is based on the RoBERTa architecture.
- It is fine-tuned to identify emotions and brand perception aspects in textual data.
- The model outputs both emotion and brand perception logits, which are converted into binary predictions.

## Model Architecture

### Dataset
- The dataset contains text from TikTok comments and captions about luxury fashion brands.
- It includes labels for both emotions and brand perception aspects.
  
### Model
- The model uses the RoBERTa architecture with two separate classifiers for emotions and brand perception aspects.
- The loss function is binary cross-entropy with logits.

### Model Training
- The model is trained using PyTorch Lightning.
- The trainer utilizes GPU acceleration if available and logs key metrics during training and validation.

## Installation
To set up the environment for this project, follow these steps:
1) Ensure you have Python installed on your system.
2) Create a virtual environment named 488env by running python -m venv 488env in your terminal.
3) Activate the virtual environment:
  - On Windows, use 488env\Scripts\activate.
  - On Unix or MacOS, use source 488env/bin/activate.
4) Install the required dependencies by running pip install -r requirements.txt.
