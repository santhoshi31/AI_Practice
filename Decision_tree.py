import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.read_csv("Loan_data.csv")
print(data.head())
print(data.shape)
data["Dependents"] = data["Dependents"].replace({"3+":3})
data["Self_Employed"] = data["Self_Employed"].replace({"No":1,"Yes":0})
data["Loan_Status"] = data["Loan_Status"].replace({"N":0,"Y":1})
data["Gender"] = data["Gender"].replace({"Male":0,"Female":1})


data["Gender"].fillna(data["Gender"].median(),inplace=True)
data["Self_Employed"].fillna(data["Self_Employed"].median(),inplace=True)
data["Dependents"].fillna(data["Dependents"].median(),inplace=True)
data["LoanAmount"].fillna(data["LoanAmount"].median(),inplace=True)
data["Credit_History"].fillna(data["Credit_History"].median(),inplace=True)
data["Loan_Amount_Term"].fillna(data["Loan_Amount_Term"].median(),inplace=True)


print(data.isnull().sum())

data["TotalIncome"] = data["ApplicantIncome"] + data["CoapplicantIncome"]
data["Income_Loan_Ratio"] = data["TotalIncome"] / data["LoanAmount"]

x =  data[["Dependents", "ApplicantIncome", "CoapplicantIncome",
     "LoanAmount", "Credit_History",
     "TotalIncome", "Income_Loan_Ratio"]]
y = data["Loan_Status"]

x_train, x_test, y_train, y_test = train_test_split(x,y, 
                            test_size=0.2, random_state=42)
model = DecisionTreeClassifier(max_depth=3,random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print("Accuracy",accuracy_score(y_test, y_pred)*100)
print("Confusion",confusion_matrix(y_test, y_pred))
print("Train Accuracy:", model.score(x_train, y_train))
print("Test Accuracy:", model.score(x_test, y_test))

