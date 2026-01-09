import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = {
    "text": [
        "Win money now",
        "Free lottery ticket",
        "Claim your free prize",
        "Hello how are you",
        "Meeting at 5pm",
        "Let's have lunch",
        "Earn cash instantly",
        "Project discussion tomorrow"
    ],
    "label": [1,1,1,0,0,0,1,0]
}


df = pd.DataFrame(data)


x_train, x_test, y_train, y_test = train_test_split(df["text"],df["label"], test_size=0.2, random_state=42)

vector = TfidfVectorizer()
x_train_vec = vector.fit_transform(x_train)
x_test_vec = vector.transform(x_test)

model = LogisticRegression()
model.fit(x_train_vec, y_train)

pred = model.predict(x_test_vec)

print("Accuracy:", accuracy_score(y_test, pred)*100)