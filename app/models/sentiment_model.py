import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the sentiment analysis model
model = BertForSequenceClassification.from_pretrained('D:/mental_health_project/saved_model_sentiment')
tokenizer = BertTokenizer.from_pretrained('D:/mental_health_project/saved_model_sentiment')

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    sentiment = torch.argmax(outputs.logits).item()
    return sentiment
