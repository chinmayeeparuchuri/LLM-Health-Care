import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import umap
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os

# Load dataset
file_path = "/Users/chinmayee/Documents/tweet-classification/data/tweet_dataset.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")

# Remove duplicates and missing values
df.drop_duplicates(subset=["Text"], inplace=True)
df.dropna(subset=["Text"], inplace=True)

# Map labels for Emotion Categorization (6 classes)
emotion_mapping = {
    "COEA": 0, "CODI": 1, "COFE": 2, "COHA": 3, "COSU": 4, "COSA": 5,
    "NOEA": 0, "NODI": 1, "NOFE": 2, "NOHA": 3, "NOSU": 4, "NOSA": 5
}

df = df[df["subtask_c"].notna()]
df["label"] = df["subtask_c"].map(emotion_mapping)

# Train-Test Split (80%-20%)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["Text"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42, stratify=df["label"]
)

# Use TinyBERT for fast training
model_name = "huawei-noah/TinyBERT_General_4L_312D"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize dataset
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_dict({
    "input_ids": train_encodings["input_ids"], 
    "attention_mask": train_encodings["attention_mask"], 
    "labels": train_labels
})

val_dataset = Dataset.from_dict({
    "input_ids": val_encodings["input_ids"], 
    "attention_mask": val_encodings["attention_mask"], 
    "labels": val_labels
})

# Load TinyBERT Model (for 6 emotion classes)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)

# Set output directory inside the script folder
output_dir = "./tinybert_emotion_model"
os.makedirs(output_dir, exist_ok=True)

# Training Arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the Model
trainer.train()

# Save Model & Tokenizer in its own folder
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print("TinyBERT Level 3 Model training completed and saved.")

# Make Predictions on Validation Set
predictions = trainer.predict(val_dataset)
preds = np.argmax(predictions.predictions, axis=1)
true_labels = val_labels

# Compute Metrics
accuracy = accuracy_score(true_labels, preds)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average="weighted")

print(f"Accuracy: {accuracy:.4f}")
print(f"Weighted Precision: {precision:.4f}")
print(f"Weighted Recall: {recall:.4f}")
print(f"Weighted F1 Score: {f1:.4f}")

# Set Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Function to Extract Embeddings
def get_embeddings(dataloader, model):
    model.to(device)  # Move model to MPS or CPU
    model.eval()  # Set model to evaluation mode
    embeddings = []
    labels = []

    for batch in dataloader:
        inputs = {key: torch.tensor(val).to(device) for key, val in batch.items() if key != 'labels'}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # Last hidden layer
            pooled_embedding = torch.mean(hidden_states, dim=1).cpu().numpy()  # Move to CPU for NumPy
            embeddings.append(pooled_embedding)
            labels.extend(batch['labels'].cpu().numpy())

    return np.vstack(embeddings), np.array(labels)

# Get Train Embeddings
train_embeddings, train_labels = get_embeddings(trainer.get_train_dataloader(), model)

# Normalize & Reduce Dimensions
scaler = StandardScaler()
train_embeddings_scaled = scaler.fit_transform(train_embeddings)

umap_model = umap.UMAP(n_components=2, random_state=42)
umap_embeddings = umap_model.fit_transform(train_embeddings_scaled)

# Plot UMAP Embeddings
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=umap_embeddings[:, 0], y=umap_embeddings[:, 1],
    hue=train_labels, palette="coolwarm", alpha=0.7
)
plt.title("UMAP Visualization of Tweet Embeddings (Level 3)")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend(title="Label", labels=["Anger", "Disgust", "Fear", "Happiness", "Surprise", "Sadness"])
plt.show()
