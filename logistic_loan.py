import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("Loan_data.csv")
print(data.head())
print(data.shape)

print(data.isnull().sum())
data["LoanAmount"].fillna(data["LoanAmount"].median(),inplace=True)
data["Credit_History"].fillna(data["Credit_History"].median(), inplace=True)
data["Dependents"] = data["Dependents"].replace("3+",3)
data["Dependents"].fillna(data["Dependents"].median(), inplace=True)
data["Loan_Status"] = data["Loan_Status"].replace({"Y":1,"N":0})
data["TotalIncome"] = data["ApplicantIncome"]+data["CoapplicantIncome"]
data["Income_Loan_Ratio"] = data["TotalIncome"] / data["LoanAmount"]
print(data.info())

x = data[["Dependents","ApplicantIncome","CoapplicantIncome","LoanAmount","Credit_History"
          ,"TotalIncome","Income_Loan_Ratio"]]
y = data["Loan_Status"]

scaler = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,random_state=42)


x_train_s = scaler.fit_transform(x_train)
x_test_s = scaler.transform(x_test)

model = LogisticRegression(max_iter=1000,class_weight="balanced")
model.fit(x_train_s, y_train)
y_pred = model.predict(x_test_s)
print("Accuracy:",accuracy_score(y_test, y_pred)*100)
print(confusion_matrix(y_test, y_pred))