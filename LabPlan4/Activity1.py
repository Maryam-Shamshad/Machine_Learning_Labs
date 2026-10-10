import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# 1. Load dataset (URL on first run, cached locally afterwards)
URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
if os.path.exists("creditcard.csv"):
    data = pd.read_csv("creditcard.csv")
else:
    data = pd.read_csv(URL)
    data.to_csv("creditcard.csv", index=False)

print("Shape:", data.shape)
print("Fraud ratio:", round(data["Class"].mean() * 100, 3), "%")

# 2. Features and target
X = data.drop("Class", axis=1)
y = data["Class"]

# 3. Train/test split (stratified keeps the fraud ratio)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Smaller training sample for speed (25% of the training set)
X_sample, _, Y_sample, _ = train_test_split(
    X_train, Y_train, train_size=0.25, random_state=42, stratify=Y_train
)

# 5. Smaller STRATIFIED test set for faster, fair evaluation
X_test_small, _, Y_test_small, _ = train_test_split(
    X_test, Y_test, train_size=20000, random_state=42, stratify=Y_test
)
print("Frauds in test subset:", int(Y_test_small.sum()))

# 6. Pipeline: scale -> SMOTE -> KNN
knn = Pipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("classifier", KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
])

# 7. Train with k=5
knn.fit(X_sample, Y_sample)
Y_Pred = knn.predict(X_test_small)

print("\nKNN Results (k=5)")
print("-------------------------------")
print("Accuracy:", round(accuracy_score(Y_test_small, Y_Pred), 4))
print("F1-Score:", round(f1_score(Y_test_small, Y_Pred), 4))

# 8. Tune k with GridSearchCV
param_grid = {"classifier__n_neighbors": [3, 5, 7]}

grid_search = GridSearchCV(
    knn,
    param_grid,
    cv=3,
    scoring="f1",
    n_jobs=1,
    verbose=1,
)
grid_search.fit(X_sample, Y_sample)

best_knn = grid_search.best_estimator_
Y_Pred_Best = best_knn.predict(X_test_small)

print("\nTuned KNN Results")
print("--------------------------------")
print("Best k:", grid_search.best_params_["classifier__n_neighbors"])
print("Accuracy:", round(accuracy_score(Y_test_small, Y_Pred_Best), 4))
print("F1-Score:", round(f1_score(Y_test_small, Y_Pred_Best), 4))

print("\nClassification report (tuned):")
print(classification_report(Y_test_small, Y_Pred_Best, digits=4))