import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import threading
from itertools import cycle, islice
import matplotlib.pyplot as plt

urls = [
    "https://www.bbc.com/news",
    "https://www.cnn.com/world",
    "https://www.reuters.com/world/",
    "https://apnews.com/"
]

headlines = []

def scrape(url):
    r = requests.get(url, timeout=15)
    s = BeautifulSoup(r.text, "html.parser")
    for h in s.find_all(["h2", "h3"]):
        t = h.get_text(strip=True)
        if t and len(t) > 20:
            headlines.append(t)

threads = []
for u in urls:
    t = threading.Thread(target=scrape, args=(u,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

if len(headlines) < 10:
    with open("headlines.json", "r", encoding="utf-8") as f:
        extra = json.load(f)
    headlines.extend(extra)

df = pd.DataFrame(headlines, columns=["headline"])
df["sentiment"] = list(islice(cycle(["positive", "negative", "neutral"]), len(df)))

X = df["headline"]
y = df["sentiment"]

vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

labels_count = pd.Series(y_pred).value_counts()
print("\nSentiment distribution:")
print(labels_count)

plt.figure(figsize=(6,6))
plt.pie(labels_count, labels=labels_count.index, autopct='%1.1f%%', startangle=90)
plt.title("Sentiment Distribution of News Headlines")
plt.savefig("sentiment_pie.png", dpi=220, bbox_inches="tight")
plt.show()
