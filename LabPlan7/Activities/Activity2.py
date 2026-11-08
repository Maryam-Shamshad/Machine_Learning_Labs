import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')
telco = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

print("Shape:", telco.shape)
print(telco.head(10))
print("Missing values:")
print(telco.isnull().sum())
print("Churn distribution:")
print(telco['Churn'].value_counts())
print(telco['Churn'].value_counts(normalize=True).round(3))
sns.countplot(data=telco, x='Churn')
plt.show()

# blanks become NaN when converting to numbers
telco['TotalCharges'] = pd.to_numeric(telco['TotalCharges'], errors='coerce')

print(telco['TotalCharges'].isnull().sum(), "blank values")
print(telco[telco['TotalCharges'].isnull()][['tenure', 'MonthlyCharges', 'TotalCharges']])

telco['TotalCharges'] = telco['TotalCharges'].fillna(0)

telco = telco.drop('customerID', axis=1)
telco['Churn'] = telco['Churn'].map({'Yes': 1, 'No': 0})

telco = pd.get_dummies(telco, drop_first=True)

X = telco.drop('Churn', axis=1)
y = telco['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(X_train.shape, X_test.shape)

gb = GradientBoostingClassifier(random_state=42)
ada = AdaBoostClassifier(random_state=42)
xgb_default = XGBClassifier(random_state=42)

gb.fit(X_train, y_train)
ada.fit(X_train, y_train)
xgb_default.fit(X_train, y_train)

for name, m in [('Gradient Boosting: ', gb), ('AdaBoost: ', ada), ('XGBoost: ', xgb_default)]:
    print(name, m.score(X_test, y_test))
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
    'subsample': [0.8, 1.0],
}

grid = GridSearchCV(XGBClassifier(random_state=42), param_grid,
                    cv=5, scoring='f1', n_jobs=-1)
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)
print("Best F1 score:", grid.best_score_)
xgb_tuned = grid.best_estimator_

rows = []
for name, m in [('XGBoost default', xgb_default), ('XGBoost tuned', xgb_tuned)]:
    pred = m.predict(X_test)
    proba = m.predict_proba(X_test)[:, 1]
    rows.append({'Model': name,
                 'Accuracy': accuracy_score(y_test, pred),
                 'Precision': precision_score(y_test, pred),
                 'Recall': recall_score(y_test, pred),
                 'F1': f1_score(y_test, pred),
                 'ROC-AUC': roc_auc_score(y_test, proba)})
print(pd.DataFrame(rows).set_index('Model').round(4))

print("Accuracy of always predicting 'No churn':", round(1 - y_test.mean(), 4))
importances = pd.Series(xgb_tuned.feature_importances_, index=X_train.columns)
importances.nlargest(5).sort_values().plot(kind='barh', figsize=(8, 4))
plt.title('Top 5 features (tuned XGBoost)')
plt.show()