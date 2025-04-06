import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# File path
file_path = "/Users/chinmayee/Documents/tweet-classification/data/tweet_dataset.xlsx"

# Load dataset
df = pd.read_excel(file_path, sheet_name="Sheet1")

# Remove unnecessary columns
df = df[['Text', 'subtask_a']]  # Keeping only the relevant columns

# Convert text to lowercase
df['Text'] = df['Text'].str.lower()

# Remove special characters, links, numbers
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.strip()  # Remove leading/trailing spaces
    return text

df['Text'] = df['Text'].apply(clean_text)

# Remove stopwords
stop_words = set(stopwords.words('english'))
df['Text'] = df['Text'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words]))

# Lemmatization
lemmatizer = WordNetLemmatizer()
df['Text'] = df['Text'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(x)]))

# Save cleaned data
df.to_csv("/Users/chinmayee/Documents/tweet-classification/data/cleaned_tweets.csv", index=False)

print(" Data Preprocessing Completed! Cleaned data saved.")
