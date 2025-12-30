import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("StudentsPerformance.csv")
print(data.head())
data["test preparation course"] = data["test preparation course"].replace({"none":0, "completed":1})
x_pd = data[["test preparation course","reading score","writing score"]]
y_pd = data["math score"]
x = x_pd.values
y = y_pd.values

x_mean = x.mean(axis=0)
x_std = x.std(axis=0)

x_scaled = (x - x_mean) / x_std
x= x_scaled


n_sample, n_feature = x.shape
w = np.zeros(n_feature)
b=0
learning_rate = 0.01
epochs = 2000


for i in range(epochs):
    y_pred = np.dot(x, w) + b
    loss = (1/n_sample)*np.sum((y-y_pred)**2)
    dw = (-2 / n_sample) * np.dot(x.T, (y - y_pred))
    db = (-2 / n_sample) * np.sum(y - y_pred)
    
    # Update parameters
    w = w - learning_rate * dw
    b = b - learning_rate * db
    
    if i % 100 == 0:
        print(f"Epoch {i} | Loss: {loss}")
print("Final Weight",w)
print("Final bies",b)
sample = np.array([1,70,7])
sample_scaled = (sample - x_mean) / x_std
prediction = np.dot(sample_scaled,w)+b
print("Prediction",prediction)
