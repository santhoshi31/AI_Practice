import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

Data = pd.read_csv("StudentsPerformance.csv")
print(Data.shape)
print(Data.describe())
print(Data.info())
print(Data["test preparation course"])
plt.bar(Data["test preparation course"], Data["math score"])
plt.xlabel("test preparation course")
plt.ylabel("math score")
plt.title("Course status vs Math Marks")
plt.show()
Data["test preparation course"] = Data['test preparation course'].replace({"none":0,"completed":1})

print(Data["test preparation course"])

x = Data[["reading score", "writing score","test preparation course"]]
y = Data["math score"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
print(y_predict)

mae = mean_absolute_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print("MAE:", mae)
print("R2 Score:", r2*100)
