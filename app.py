from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

sentiment_pipeline = pipeline("sentiment-analysis")

@app.route("/")
def home():
    return "AI Sentiment Analyzer API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"})

    result = sentiment_pipeline(text)[0]

    return jsonify({
        "text": text,
        "sentiment": result["label"],
        "confidence": round(result["score"], 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
