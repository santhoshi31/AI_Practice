import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

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
data["TotalIncome"] = data["ApplicantIncome"] + data["CoapplicantIncome"]
data["Income_Loan_Ratio"] = data["TotalIncome"] / data["LoanAmount"]

print(data.isnull().sum())

x = data[["ApplicantIncome","CoapplicantIncome","Dependents","LoanAmount","Credit_History",
          "TotalIncome","Income_Loan_Ratio"]]

y = data["Loan_Status"]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.1,random_state=42)

Random_Forest = RandomForestClassifier(max_depth=2,min_samples_split= 0.01,\
                                            max_features= 0.8,
                                            max_samples= 0.8,
                               random_state=42,class_weight="balanced")
model = Random_Forest.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("Accuracy",accuracy_score(y_test, y_pred)*100)
print("Score",confusion_matrix(y_test, y_pred))

