import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

# ✅ Correct file path
file_path = "/Users/chinmayee/Documents/tweet-classification/data/tweet_dataset.xlsx"

# ✅ Load dataset from Sheet1
df = pd.read_excel(file_path, sheet_name="Sheet1")

# ✅ Drop duplicates and missing values
df.drop_duplicates(subset=["Text"], inplace=True)
df.dropna(subset=["Text"], inplace=True)

# ✅ Filter only "Sadness" category for Level 4
df = df[df["subtask_c"] == "COSA"]  # Only sadness tweets

# ✅ Label Mapping for Depression vs. Non-Depression
label_mapping = {"CODE": 1, "COND": 0}  # 1 = Depression, 0 = Non-Depression
df = df[df["subtask_d"].notna()]
df["label"] = df["subtask_d"].map(label_mapping)

# ✅ Train-Test Split (Stratified)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["Text"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42, stratify=df["label"]
)

# 🚀 Use TinyBERT (Fast & Efficient)
model_name = "huawei-noah/TinyBERT_General_4L_312D"
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

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

# ✅ Load TinyBERT Model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# ✅ Training Arguments (Faster Execution)
training_args = TrainingArguments(
    output_dir="./tinybert_depression_model",  # Saves in the current folder
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=16,
    num_train_epochs=1,  # Minimal epochs for speed
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

# ✅ Train Model (FAST)
trainer.train()

# ✅ Save Model & Tokenizer in the same folder
trainer.save_model("./tinybert_depression_model")
tokenizer.save_pretrained("./tinybert_depression_model")

print("✅ TinyBERT Model training completed and saved!")

# ✅ Make Predictions on Validation Set
predictions = trainer.predict(val_dataset)
preds = np.argmax(predictions.predictions, axis=1)
true_labels = val_labels

# ✅ Compute Metrics
accuracy = accuracy_score(true_labels, preds)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average="weighted")

# ✅ Display Confusion Matrix
conf_matrix = confusion_matrix(true_labels, preds)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Depression", "Depression"], yticklabels=["Non-Depression", "Depression"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for Depression Classification")
plt.show()

print(f"✅ Accuracy: {accuracy:.4f}")
print(f"✅ Weighted Precision: {precision:.4f}")
print(f"✅ Weighted Recall: {recall:.4f}")
print(f"✅ Weighted F1 Score: {f1:.4f}")
print(f"✅ Confusion Matrix:\n{conf_matrix}")

# ✅ Embedding Visualization
def get_embeddings(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model.bert(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()  # Extract CLS token embeddings

train_embeddings = get_embeddings(train_texts)
val_embeddings = get_embeddings(val_texts)

# ✅ PCA for Dimensionality Reduction
pca = PCA(n_components=2)
train_pca = pca.fit_transform(train_embeddings)
val_pca = pca.transform(val_embeddings)

# ✅ Scatter Plot for Visualization
plt.figure(figsize=(8, 6))
sns.scatterplot(x=train_pca[:, 0], y=train_pca[:, 1], hue=train_labels, palette=["blue", "red"], alpha=0.7)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("TinyBERT Embedding Visualization for Depression Classification")
plt.legend(["Non-Depression", "Depression"])
plt.show()