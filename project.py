import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import threading

urls = [
    "https://www.bbc.com/news",
    "https://www.cnn.com/world"
]

headlines = []

def scrape(url):
    r = requests.get(url)
    s = BeautifulSoup(r.text, "html.parser")
    for h in s.find_all("h3"):
        text = h.get_text()
        if text:
            headlines.append(text)

threads = []
for u in urls:
    t = threading.Thread(target=scrape, args=(u,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

with open("headlines.json", "w", encoding="utf-8") as f:
    json.dump(headlines, f, ensure_ascii=False, indent=2)

with open("headlines.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data, columns=["headline"])
df["sentiment"] = ["positive", "negative", "neutral"] * (len(df)//3)

X = df["headline"]
y = df["sentiment"]

vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
