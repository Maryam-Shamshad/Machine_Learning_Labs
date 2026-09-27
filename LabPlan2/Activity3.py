# Maryam Shamshad
# 23-NTU-CS-1047
# Activity 3: Linear and Polynomial Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics

# Data Loading and Splitting
data = pd.read_csv("Car Price Prediction.csv")
X = data[['wheelbase', 'curbweight', 'enginesize', 'horsepower']]
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluate Degrees and Polynomial Models
for degree in [1, 2, 3, 4, 5]:
    poly = PolynomialFeatures(degree=degree)
    X_tr = poly.fit_transform(X_train) if degree > 1 else X_train
    X_te = poly.transform(X_test) if degree > 1 else X_test
    model = LinearRegression().fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    r2 = metrics.r2_score(y_test, y_pred)
    print(f"Degree {degree} , RMSE: {rmse:.2f} | R2: {r2:.4f}")

# Ploting Curves
sorted_data = data[['horsepower', 'price']].sort_values(by='horsepower')
x_vis = sorted_data[['horsepower']]
y_vis = sorted_data['price']
plt.figure(figsize=(8, 6))
plt.scatter(x_vis, y_vis, color='blue', alpha=0.5, label='Actual Data')
for degree, color in zip([2, 3], ['green', 'orange']):
    X_poly = PolynomialFeatures(degree=degree).fit_transform(x_vis)
    y_curve = LinearRegression().fit(X_poly, y_vis).predict(X_poly)
    plt.plot(x_vis, y_curve, color=color, linewidth=2, label=f'Degree {degree}')
plt.xlabel('horsepower')
plt.ylabel('price')
plt.title('Polynomial Regression Fits')
plt.legend()
plt.show()