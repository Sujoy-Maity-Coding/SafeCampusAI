from flask import Flask, request, jsonify
import pickle
import re
import random

app = Flask(__name__)

# Load models
category_model = pickle.load(
    open("category_model.pkl", "rb")
)

priority_model = pickle.load(
    open("priority_model.pkl", "rb")
)

tfidf = pickle.load(
    open("tfidf.pkl", "rb")
)

# Clean text
def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z\\s]', '', text)

    return text


@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    complaint = data["text"]

    clean = clean_text(complaint)

    vector = tfidf.transform([clean])

    # Predict category
    category = category_model.predict(vector)[0]

    # Predict priority
    priority = priority_model.predict(vector)[0]

    # Dynamic risk score
    if priority == "Critical":
        risk_score = random.randint(91, 100)

    elif priority == "High":
        risk_score = random.randint(71, 90)

    elif priority == "Medium":
        risk_score = random.randint(31, 70)

    else:
        risk_score = random.randint(1, 30)

    return jsonify({

        "category": category,

        "priority": priority,

        "risk_score": risk_score
    })


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )