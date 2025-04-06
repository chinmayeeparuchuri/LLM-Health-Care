import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load Model & Tokenizer
model_path = "../models/tinybert_emotion_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    return "Emotional" if prediction == 1 else "Non-Emotional"

# Test Inference
sample_text = "The pandemic made me feel so hopeless and scared."
print("Prediction:", predict_emotion(sample_text))
