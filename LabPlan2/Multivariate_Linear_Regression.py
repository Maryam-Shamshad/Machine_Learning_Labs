# Maryam Shamshad
# 23-NTU-CS-1047
# Multivariate Linear Regression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

df = pd.read_csv("Admission_Predict.csv")
df.drop("Serial No.", axis=1, inplace=True)
y = df["Chance of Admit "]
df.drop("Chance of Admit ", axis=1, inplace=True)
print(df.head())

simple_lr = LinearRegression()
simple_lr.fit(df[["GRE Score"]], y)
simple_pred = simple_lr.predict(df[["GRE Score"]])

plt.scatter(df["GRE Score"],y,color="green",label="Actual Data",alpha=0.5)
plt.plot(df["GRE Score"],simple_pred,color="red",linewidth=3,label="Regression Line")
plt.xlabel("GRE Score")
plt.ylabel("Admission Chance")
plt.title("GRE Score vs Admission Chance")
plt.legend()
plt.show()

x_train, x_test, y_train, y_test = train_test_split(df,y,test_size=0.2,random_state=42)
lr = LinearRegression()
lr.fit(x_train, y_train)
pred = lr.predict(x_test)

rmse = np.sqrt(metrics.mean_squared_error(y_test, pred))
print("RMSE:", rmse)

r2 = metrics.r2_score(y_test, pred)
print("R2 Score:", r2)

coefficients = lr.coef_
features = df.columns
plt.figure(figsize=(8, 5))
plt.barh(features,coefficients,color="teal")

plt.xlabel("Coefficient Value (Weight)")
plt.title("Feature Importance in Multivariate Linear Regression")
plt.axvline(x=0,color="black",linewidth=0.8)

plt.show()