import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix

data = pd.read_csv("spam.csv", encoding="latin-1")[["v1","v2"]]
data.columns = ["label","message"]
print(data.head())
print(data["label"].value_counts())

data["label"] = data["label"].replace({"spam":1,"ham":0})

vectore = TfidfVectorizer(lowercase=True, stop_words="english", max_features=3000)

x = vectore.fit_transform(data["message"])
y = data["label"]

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=42, 
                                                    test_size=0.2, stratify=y)
model = LogisticRegression(max_iter=1000,
                           class_weight="balanced")

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print("Accuracy", accuracy_score(y_test, y_pred)*100)
print("Confusion",confusion_matrix(y_test, y_pred))