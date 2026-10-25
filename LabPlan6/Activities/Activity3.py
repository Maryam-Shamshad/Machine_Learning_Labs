import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

house = pd.read_csv("train.csv")
print("Shape:", house.shape)
print("\nFirst 10 rows:")
print(house.head(10))
missing = house.isnull().sum()
print("\nMissing values (columns that have missing values):")
print(missing[missing > 0].sort_values(ascending=False))
print("\nSummary statistics of SalePrice:")
print(house["SalePrice"].describe())
X = house.drop(columns=["Id", "SalePrice"])
y = house["SalePrice"]
print("X shape:", X.shape)
print("y shape:", y.shape)
num_cols = X.select_dtypes(include=np.number).columns
cat_cols = X.select_dtypes(exclude=np.number).columns

for col in num_cols:
    X[col] = X[col].fillna(X[col].median())
for col in cat_cols:
    X[col] = X[col].fillna("None")
X = pd.get_dummies(X, columns=cat_cols, dtype=int)
print("Missing values left:", X.isnull().sum().sum())
print("Shape after preprocessing:", X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
default_rf = RandomForestRegressor(random_state=42)   # default parameters
default_rf.fit(X_train, y_train)

default_pred = default_rf.predict(X_test)
default_mae = mean_absolute_error(y_test, default_pred)
default_mse = mean_squared_error(y_test, default_pred)
default_rmse = np.sqrt(default_mse)
default_r2 = r2_score(y_test, default_pred)
print("Default model (test set)")
print("MAE :", round(default_mae, 2))
print("MSE :", round(default_mse, 2))
print("RMSE:", round(default_rmse, 2))
print("R2  :", round(default_r2, 4))
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": [0.7, 1.0]
}
grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
print("Best hyperparameters:", grid_search.best_params_)
print("Best cross-validation score (negative MSE):", grid_search.best_score_)
cv_mse = -grid_search.best_score_
cv_rmse = np.sqrt(cv_mse)
print("Cross-validation MSE :", round(cv_mse, 2))
print("Cross-validation RMSE:", round(cv_rmse, 2))
best_rf = grid_search.best_estimator_
best_pred = best_rf.predict(X_test)
best_mae = mean_absolute_error(y_test, best_pred)
best_mse = mean_squared_error(y_test, best_pred)
best_rmse = np.sqrt(best_mse)
best_r2 = r2_score(y_test, best_pred)
print("Optimized model (test set)")
print("MAE :", round(best_mae, 2))
print("MSE :", round(best_mse, 2))
print("RMSE:", round(best_rmse, 2))
print("R2  :", round(best_r2, 4))
results = pd.DataFrame({
    "Metric": ["MAE", "MSE", "RMSE", "R2"],
    "Default Model": [default_mae, default_mse, default_rmse, default_r2],
    "Optimized Model": [best_mae, best_mse, best_rmse, best_r2]
})
print(results.to_string(index=False))
plt.figure(figsize=(7, 6))
plt.scatter(y_test, best_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color="red", linestyle="--")          # perfect prediction line
plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")
plt.title("Actual vs Predicted House Prices")
plt.grid(alpha=0.3)
plt.show()