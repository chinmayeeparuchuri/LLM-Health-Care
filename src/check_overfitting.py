import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Correct file path
file_path = "/Users/chinmayee/Documents/tweet-classification/data/tweet_dataset.xlsx"

# Load dataset from Sheet1
df = pd.read_excel(file_path, sheet_name="Sheet1")

# Normalize Column Names to Lowercase
df.columns = df.columns.str.lower().str.strip()  # Convert all column names to lowercase
print("Columns in dataset:", df.columns)  # Debugging step

if "text" not in df.columns:
    raise KeyError(" Column 'text' not found! Check CSV file and use correct column name.")

# Extract TF-IDF Features
def extract_tfidf_features(max_feat=3000):  
    vectorizer = TfidfVectorizer(max_features=max_feat)
    X = vectorizer.fit_transform(df["text"])  # Column is now guaranteed to be lowercase
    return X, vectorizer

X, vectorizer = extract_tfidf_features(3000)  
y = df["subtask_a"].factorize()[0]

# Save TF-IDF Vectorizer as Pickle File
with open("tfidf_vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)
print(" TF-IDF Vectorizer saved as tfidf_vectorizer.pkl")

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train with Different Regularization Strengths
for C_value in [0.5, 0.1]:  
    print(f"\n Training Logistic Regression with C={C_value}")

    model = LogisticRegression(C=C_value, max_iter=500)
    model.fit(X_train, y_train)

    # Save Model as Pickle File
    model_filename = f"lr_model_C{C_value}.pkl"
    with open(model_filename, "wb") as file:
        pickle.dump(model, file)
    print(f"     Model saved as {model_filename}")

    # Check Accuracy
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    precision = precision_score(y_test, model.predict(X_test), average='weighted')
    recall = recall_score(y_test, model.predict(X_test), average='weighted')
    f1 = f1_score(y_test, model.predict(X_test), average='weighted')

    print(f"    Training Accuracy: {train_acc:.4f}")
    print(f"    Testing Accuracy: {test_acc:.4f}")
    print(f"    Weighted Precision: {precision:.4f}")
    print(f"    Weighted Recall: {recall:.4f}")
    print(f"    Weighted F1-score: {f1:.4f}")

    # Plot Learning Curve
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, train_sizes=np.linspace(0.1, 1.0, 5), scoring="accuracy"
    )

    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, 'o-', label="Training Accuracy", color="blue")
    plt.plot(train_sizes, test_mean, 'o-', label="Validation Accuracy", color="green")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.title(f"Learning Curve (C={C_value}, max_features=3000)")
    plt.legend()
    plt.grid()
    plt.show(block=True)
