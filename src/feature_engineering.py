import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
file_path = "/Users/chinmayee/Documents/tweet-classification/data/tweet_dataset.xlsx"
df = pd.read_excel(file_path)


# Print column names for verification
print("Columns in dataset:", df.columns)

# Debug: Check unique values in 'subtask_a' before mapping
print("\nUnique values in 'subtask_a' before mapping:")
print(df["subtask_a"].unique())

# Define label mapping
label_mapping = {"CORO": 1, "NOCO": 0}

# Apply mapping
df["label"] = df["subtask_a"].map(label_mapping)

# Debug: Check for unmapped values
if df["label"].isnull().sum() > 0:
    print("\nWarning: Some labels could not be mapped!")
    print(df[df["label"].isnull()][["ID", "subtask_a"]])  # Show problem rows
    raise ValueError("Error: Unmapped values detected in 'subtask_a'. Fix dataset.")

# Convert to integer
df["label"] = df["label"].astype(int)


# Preprocessing: Remove NaN text entries
df.dropna(subset=["Text"], inplace=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df["Text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model (RandomForestClassifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the model and vectorizer
import pickle
output_model_path = "/Users/chinmayee/Documents/tweet-classification/models/random_forest_model.pkl"
output_vectorizer_path = "/Users/chinmayee/Documents/tweet-classification/models/tfidf_vectorizer.pkl"

os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
with open(output_model_path, "wb") as model_file:
    pickle.dump(model, model_file)
with open(output_vectorizer_path, "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("\nModel and vectorizer saved successfully.")
