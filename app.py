from flask import Flask, request, jsonify
from flask_cors import CORS
from textblob import TextBlob

import pickle
import random
import re

app = Flask(__name__)
CORS(app)

# LOAD TRAINED MODELS
model = pickle.load(
    open("category_model.pkl", "rb")
)

modelPriority = pickle.load(
    open("priority_model.pkl", "rb")
)

tfidf = pickle.load(
    open("tfidf.pkl", "rb")
)

# CLEAN TEXT
def clean_text(text):

    text = text.lower()

    text = re.sub(
        r'[^a-zA-Z\s]',
        '',
        text
    )

    return text


# EMOTION + AUTHENTICITY DETECTION
def detect_emotion_and_authenticity(text):

    blob = TextBlob(text)

    polarity = blob.sentiment.polarity

    # EMOTION DETECTION
    if polarity < -0.5:

        emotion = "Fear/Panic"

    elif polarity < 0:

        emotion = "Stress"

    elif polarity > 0.5:

        emotion = "Neutral"

    else:

        emotion = "Concern"

    # DANGER WORDS
    danger_words = [

        "help",
        "attack",
        "threat",
        "harass",
        "ragging",
        "beating",
        "unsafe",
        "violence",
        "abuse",
        "panic",
        "fight",
        "blood"
    ]

    score = 0

    for word in danger_words:

        if word in text.lower():

            score += 12

    # ADD POLARITY IMPACT
    score += abs(polarity) * 40

    authenticity_score = min(
        round(score),
        100
    )

    # AUTHENTICITY RESULT
    if authenticity_score < 20:

        authenticity = "Potential Fake"

    elif authenticity_score < 50:

        authenticity = "Possibly Genuine"

    else:

        authenticity = "Likely Genuine"

    return (
        emotion,
        authenticity,
        authenticity_score
    )


@app.route('/predict', methods=['POST'])
def predict():

    try:

        data = request.json

        description = data['description']

        # CLEAN TEXT
        sample_clean = [
            clean_text(description)
        ]

        # TF-IDF VECTORIZE
        sample_vector = tfidf.transform(
            sample_clean
        )

        # CATEGORY PREDICTION
        category_prediction = model.predict(
            sample_vector
        )

        # PRIORITY PREDICTION
        priority_prediction = modelPriority.predict(
            sample_vector
        )

        priority = priority_prediction[0]

        # DYNAMIC RISK SCORE
        if priority == "Critical":

            risk_score = random.randint(91, 100)

        elif priority == "High":

            risk_score = random.randint(71, 90)

        elif priority == "Medium":

            risk_score = random.randint(31, 70)

        else:

            risk_score = random.randint(1, 30)

        # EMOTION + AUTHENTICITY
        emotion, authenticity, authenticity_score = \
            detect_emotion_and_authenticity(
                description
            )

        # FINAL RESPONSE
        return jsonify({

            "category":
                category_prediction[0],

            "priority":
                priority,

            "risk_score":
                risk_score,

            "emotion":
                emotion,

            "authenticity":
                authenticity,

            "authenticity_score":
                authenticity_score
        })

    except Exception as e:

        return jsonify({

            "error": str(e)

        })


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5001,
        debug=True
    )