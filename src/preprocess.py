import pandas as pd

# Correct file path
file_path = "/Users/chinmayee/Documents/tweet-classification/data/tweet_dataset.xlsx"

# Load dataset from Sheet1
df = pd.read_excel(file_path, sheet_name="Sheet1")

# Display basic info
print(df.head())
print(df.info())
