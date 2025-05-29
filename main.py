import os, re, requests
from datetime import datetime
import numpy as np
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from google.cloud import bigquery
from nltk.corpus import stopwords
import nltk
nltk.download("stopwords") 

stop_words = set(stopwords.words("english"))

API_KEY = os.getenv("878fac9d70a94e8a8f27ac7ad4151ef9")
BQ_PROJECT = os.getenv("cloud-461215")
BQ_DATASET = os.getenv("news_dataset")
BQ_TABLE = os.getenv("articles")

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)

def clean_text(text):
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    tokens = text.lower().split()
    return " ".join([t for t in tokens if t not in stop_words])

def classify_sentiment(text):
    inputs = tokenizer(text[:512], return_tensors="tf", truncation=True, padding=True)
    outputs = model(inputs)[0]
    scores = tf.nn.softmax(outputs, axis=1).numpy()[0]
    label = np.argmax(scores) + 1
    return "NEGATIVE" if label <= 2 else "NEUTRAL" if label == 3 else "POSITIVE"

def run():
    url = f"https://newsapi.org/v2/top-headlines?country=us&category=business&language=en&apiKey={API_KEY}"
    articles = requests.get(url).json().get("articles", [])
    results = []

    for a in articles:
        title, desc = a.get("title", ""), a.get("description", "")
        sentiment = classify_sentiment(clean_text(f"{title} {desc}"))
        results.append({
            "title": title,
            "description": desc,
            "published_at": a.get("publishedAt"),
            "sentiment": sentiment,
            "ingested_at": datetime.utcnow().isoformat()
        })

    client = bigquery.Client(project=BQ_PROJECT)
    table_id = f"{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}"
    errors = client.insert_rows_json(table_id, results)
    if errors:
        print("Errors:", errors)
    else:
        print("Successfully uploaded to BigQuery")

if __name__ == "__main__":
    run()