import os
import re
import requests
from datetime import datetime
import numpy as np
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from google.cloud import bigquery
from nltk.corpus import stopwords
import nltk
from flask import Flask, jsonify

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Load environment variables
API_KEY = os.getenv("NEWS_API_KEY")
BQ_PROJECT = os.getenv("BQ_PROJECT")
BQ_DATASET = os.getenv("BQ_DATASET")
BQ_TABLE = os.getenv("BQ_TABLE")

# Load sentiment model
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)

# Flask app
app = Flask(__name__)

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

def analyze_and_upload():
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
    return errors

@app.route("/", methods=["GET"])
def index():
    try:
        errors = analyze_and_upload()
        if errors:
            return jsonify({"status": "error", "details": errors}), 500
        return jsonify({"status": "success", "message": "News analyzed and uploaded to BigQuery"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
