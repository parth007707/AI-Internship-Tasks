import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC

texts = [
    "I love AI",
    "Amazing product",
    "Worst experience",
    "Bad service"
]

labels = [
    "Positive",
    "Positive",
    "Negative",
    "Negative"
]

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(texts)

model = SVC()

model.fit(X,labels)

test = vectorizer.transform(["Excellent work"])

print(model.predict(test))
