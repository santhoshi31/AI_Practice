from sklearn.feature_extraction.text import CountVectorizer


doctument = ["I love NLP",
    "I love machine learning",
    "NLP loves me"]

vector = CountVectorizer()
x = vector.fit_transform(doctument)

print("Vocabulary:", vector.get_feature_names_out())
print("Bow Matrix:\n", x.toarray())