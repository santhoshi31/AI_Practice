import pandas as pd
import numpy as np
from sklearn.datasets import load_wine

wine = load_wine()
x = wine.data
y = wine.target
print(x.shape)
print(y.shape)
print(np.unique(y))
x_mean = x.mean(axis=0)
x_std = x.std(axis=0)


x_scaled = (x - x_mean) / x_std
x = x_scaled

np.random.seed(42)
ind = np.random.permutation(len(x))
split = int(0.8 * len(x))

train_idx = ind[:split]
test_idx = ind[split:]

x_train, x_test = x[train_idx], x[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def knn_predict(X_train, y_train, x_test, k=5):
    distances = []

    for i in range(len(X_train)):
        dist = euclidean_distance(X_train[i], x_test)
        distances.append((dist, y_train[i]))

    distances.sort(key=lambda x: x[0])

    k_nearest = distances[:k]
    labels = [label for _, label in k_nearest]

    prediction = max(set(labels), key=labels.count)
    return prediction
y_pred = []

for x in x_test:
    pred = knn_predict(x_train, y_train, x, k=5)
    y_pred.append(pred)

y_pred = np.array(y_pred)

accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy*100)

