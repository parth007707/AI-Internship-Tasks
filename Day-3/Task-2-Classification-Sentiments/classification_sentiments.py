import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Sample dataset
data = pd.DataFrame({
    "text": [
        "I love this product",
        "This is amazing",
        "Very bad experience",
        "I hate this service",
        "Excellent quality",
        "Worst purchase ever"
    ],
    "sentiment": [
        "Positive",
        "Positive",
        "Negative",
        "Negative",
        "Positive",
        "Negative"
    ]
})

# Train model
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

model.fit(data["text"], data["sentiment"])

# Test samples
test_sentences = [
    "I really love it",
    "This is terrible",
    "Fantastic experience",
    "Very disappointing"
]

predictions = model.predict(test_sentences)

print("Sentiment Predictions")
for text, pred in zip(test_sentences, predictions):
    print(f"{text} --> {pred}")

# Training accuracy
train_predictions = model.predict(data["text"])
accuracy = accuracy_score(data["sentiment"], train_predictions)

print("\nTraining Accuracy:", accuracy)
