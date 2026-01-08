from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "i love nlp",
    "i love machine learning",
    "nlp loves me"
]

vector = TfidfVectorizer()
x = vector.fit_transform(documents)

print("Voc", vector.get_feature_names_out())
print("TF-IDF Matrix:\n", x.toarray())