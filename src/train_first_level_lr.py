import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Correct file path
file_path = "/Users/chinmayee/Documents/tweet-classification/data/tweet_dataset.xlsx"

# Load dataset from Sheet1
df = pd.read_excel(file_path, sheet_name="Sheet1")

# Ensure the dataset contains the required columns
print("Columns in dataset:", df.columns)

# Remove rows with missing labels
df = df.dropna(subset=["Text", "subtask_a"])

# Map labels (CORO = 1, NOCO = 0)
df["subtask_a"] = df["subtask_a"].map({"CORO": 1, "NOCO": 0})

# Load TF-IDF features
with open("data/tfidf_features.pkl", "rb") as f:
    X = pickle.load(f)

y = df["subtask_a"]  # Target labels

# Ensure X and y have the same number of samples
print(f"Feature matrix shape: {X.shape}")
print(f"Label vector shape: {y.shape}")

if X.shape[0] != y.shape[0]:
    min_len = min(X.shape[0], y.shape[0])
    X = X[:min_len]
    y = y.iloc[:min_len]

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model
with open("models/first_level_lr_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ First-level classifier trained and saved successfully!")
