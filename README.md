# Hierarchical Tweet Emotion Classification - NLP Hackathon  

## Overview  
This repository contains the code and models for hierarchical tweet emotion classification, developed for an #NLP-based hackathon. The project analyzes tweets related to the COVID-19 pandemic and classifies them across four hierarchical levels to detect emotional well-being, with a focus on depression detection at the final stage.  


## Project Overview  
The goal of this project is to build a *multi-level tweet classification system* that categorizes tweets based on:  
1. *COVID-19 relevance*  
2. *Emotional vs. Non-Emotional content*  
3. *Specific emotions (Fear, Anger, Sadness, etc.)*  
4. *Sadness classification into Depression vs. Non-Depression*  

### Why Hierarchical Classification?  
Instead of training a single model to classify all categories at once, we break down the classification into structured steps, making it more:  
- *Accurate* (better separation between categories)  
- *Efficient* (each model is trained for a specific task)  
- *Interpretable* (clear classification process)  

## Dataset Structure  
The dataset consists of *71,641 tweets, labeled across **four hierarchical levels*:  

| Level | Category | Label Codes | Tweet Count |  
|-------|------------------------------|------------|--------------|  
| 1 | COVID-19 vs. Non-COVID-19 | CORO, NOCO | 47,156 (CORO), 24,485 (NOCO) |  
| 2 | Emotional vs. Non-Emotional | COEM, CONE, NOEM, NONE | See below |  
| 3 | Emotion Categorization | COEA, CODI, COFE, COHA, COSU, COSA, NOEA, NODI, NOFE, NOHA, NOSU, NOSA | See below |  
| 4 | Sadness → Depression vs. Non-Depression | CODE, COND, NODE, NOND | See below |  

## Hierarchical Classification Levels  

### Level 1: COVID-19 vs. Non-COVID-19 Classification  
- *Model Used*: Logistic Regression, Random Forest  
- *Task: Classify tweets as **COVID-19 related (CORO)* or *Non-COVID (NOCO)*  

### Level 2: Emotional vs. Non-Emotional Classification  
- *Model Used: **DistilBERT* (Fine-tuned)  
- *Task: Classify tweets as **Emotional (COEM, NOEM)* or *Non-Emotional (CONE, NONE)*  

### Level 3: Emotion Categorization  
- *Model Used: **TinyBERT* (Fine-tuned)  
- *Task: Classify emotional tweets into **Anger, Fear, Sadness, Disgust, Happiness, Surprise*  

### Level 4: Depression vs. Non-Depression Classification  
- *Model Used: **TinyBERT* (Fine-tuned)  
- *Task: Further classify **Sadness tweets (COSA, NOSA)* into *Depression (CODE, NODE)* or *Non-Depression (COND, NOND)*  

## Project File Structure  

```bash
├── data/                      # Dataset files  
│   ├── tweet_dataset.xlsx  
│   ├── cleaned_tweets.xlsx  
├── src/                       # Source code  
│   ├── preprocess.py  
│   ├── clean_data.py  
│   ├── feature_engineering.py  
│   ├── train_level2.py        # Emotional vs. Non-Emotional (DistilBERT)  
│   ├── train_level3.py        # Emotion Categorization (TinyBERT)  
│   ├── train_level4.py        # Depression Classification (TinyBERT)  
├── models/                    # Trained models  
│   ├── tfidf_vectorizer.pkl  
│   ├── lr_model_C0.1.pkl  
│   ├── lr_model_C0.5.pkl  
│   ├── random_forest_model.pkl  
│   ├── tinybert_depression_model/  
│   ├── distilbert_emotion_model/  
├── requirements.txt           # Dependencies  
├── README.md                  # Project Documentation  


```
## ⚙️ Installation and Setup  
### Clone the repository:  
```bash
git clone https://github.com/yourusername/tweet-emotion-classification.git
cd tweet-emotion-classification

```
## 🏋️‍♂️ Training and Inference
### Train Level 2 (Emotional vs. Non-Emotional Classification):
```bash
python src/train_level2.py

```
## Train Level 3 (Emotion Categorization):
```bash
python src/train_level3.py

```
## Train Level 4 (Depression Detection):
```bash
python src/train_level4.py

```
## 🔍 Run Inference
## To predict emotions in new tweets:

```bash
from transformers import pipeline  
classifier = pipeline("text-classification", model="./models/distilbert_emotion_model")  
classifier("I am feeling very scared about the future.")  

```
## 📊 Results and Visualization
📌 Confusion Matrix <br>
📌 Precision, Recall, F1-score <br>
📌 PCA Visualization of Tweet Embeddings <br>


## 🚀 Future Improvements
- Deploy as a web API using Flask/FastAPI<br>
- Improve accuracy with larger Transformer models<br>
- Implement data augmentation techniques<br>



This hackathon project was an exciting experience, allowing us to explore LLM-based NLP models and apply hierarchical classification for mental health trend analysis. Securing 4th place as finalists was a great achievement!




