import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

# Heart Disease Dataset from the UCI Repository
heart_disease = fetch_ucirepo(id=45)
X = heart_disease.data.features     
y = heart_disease.data.targets       

# Features and Target in one DataFrame
heart = X.copy()
heart["target"] = y.iloc[:, 0]
print("Shape:", heart.shape)
print("\nFirst 10 rows:")
print(heart.head(10))
print("\nMissing values:")
print(heart.isnull().sum())
print("\nTarget class distribution (before making it 0/1):")
print(heart["target"].value_counts().sort_index())
heart["target"] = (heart["target"] > 0).astype(int)

# Deal Missing Values and Fill with the Median
heart = heart.fillna(heart.median())

# Encode categorical features 
heart = pd.get_dummies(heart, columns=["cp", "restecg", "slope", "thal"],
                       drop_first=True, dtype=int)
print("\nMissing values left:", heart.isnull().sum().sum())
print("Shape after cleaning:", heart.shape)
print("Target class distribution (0/1):")
print(heart["target"].value_counts())

# Separate X and y
X = heart.drop(columns=["target"])
y = heart["target"]
print("\nX shape:", X.shape)
print("y shape:", y.shape)

# Spliting in Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Basic Random Forest
basic_rf = RandomForestClassifier(random_state=42)   
basic_rf.fit(X_train, y_train)
basic_pred = basic_rf.predict(X_test)
basic_acc = accuracy_score(y_test, basic_pred)
print("\nBasic Random Forest test accuracy:", round(basic_acc, 4))

# Parameter grid
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2"]
}

# GridSearchCV
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)
print("\nBest hyperparameters:", grid_search.best_params_)
print("Best cross-validation accuracy:", round(grid_search.best_score_, 4))

# Evaluate the best model
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
print("\nAccuracy :", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall   :", round(recall_score(y_test, y_pred), 4))
print("F1-score :", round(f1_score(y_test, y_pred), 4))
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion matrix:")
print(cm)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature importances
importances = pd.Series(best_rf.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)
plt.figure(figsize=(8, 6))
importances.sort_values().plot(kind="barh")
plt.title("Feature Importances (Best Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()
print("\nThree most influential features:")
print(importances.head(3))