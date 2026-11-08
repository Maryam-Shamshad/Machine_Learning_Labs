import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.filterwarnings('ignore')

water = pd.read_csv('water_potability.csv')

print("Shape:", water.shape)
print(water.head(10))
print("Missing values:")
print(water.isnull().sum())
print("Class distribution:")
print(water['Potability'].value_counts())
sns.countplot(data=water, x='Potability')
plt.show()

X = water.drop('Potability', axis=1)
y = water['Potability']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(len(X_train), len(X_test))
medians = X_train.median()
X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)
print(X_train.isnull().sum().sum(), X_test.isnull().sum().sum())


lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
svc = make_pipeline(StandardScaler(), SVC(probability=True, random_state=42))
rf = RandomForestClassifier(n_estimators=100, random_state=42)

models = [('lr', lr), ('svc', svc), ('rf', rf)]

voting_hard = VotingClassifier(models, voting='hard')
voting_soft = VotingClassifier(models, voting='soft')

lr.fit(X_train, y_train)
svc.fit(X_train, y_train)
rf.fit(X_train, y_train)
voting_hard.fit(X_train, y_train)
voting_soft.fit(X_train, y_train)

tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

bag_10 = BaggingClassifier(DecisionTreeClassifier(random_state=42),
                           n_estimators=10, max_samples=0.5,
                           bootstrap=True, random_state=42)
bag_100 = BaggingClassifier(DecisionTreeClassifier(random_state=42),
                            n_estimators=100, max_samples=0.5,
                            bootstrap=True, random_state=42)
paste_100 = BaggingClassifier(DecisionTreeClassifier(random_state=42),
                              n_estimators=100, max_samples=0.5,
                              bootstrap=False, random_state=42)

bag_10.fit(X_train, y_train)
bag_100.fit(X_train, y_train)
paste_100.fit(X_train, y_train)

all_models = {
    'Logistic Regression': lr,
    'SVC': svc,
    'Random Forest': rf,
    'Single Decision Tree': tree,
    'Voting (hard)': voting_hard,
    'Voting (soft)': voting_soft,
    'Bagging (10 trees)': bag_10,
    'Bagging (100 trees)': bag_100,
    'Pasting (100 trees)': paste_100,
}

rows = []
for name, model in all_models.items():
    pred = model.predict(X_test)
    rows.append({'Model': name,
                 'Accuracy': accuracy_score(y_test, pred),
                 'Precision': precision_score(y_test, pred),
                 'Recall': recall_score(y_test, pred),
                 'F1': f1_score(y_test, pred)})

results = pd.DataFrame(rows).set_index('Model').round(4)
print(results.sort_values('Accuracy', ascending=False))
from sklearn.metrics import ConfusionMatrixDisplay

best_name = results['Accuracy'].idxmax()
print("Best model:", best_name)
ConfusionMatrixDisplay.from_estimator(all_models[best_name], X_test, y_test)
plt.show()

singles = ['Logistic Regression', 'SVC', 'Random Forest', 'Single Decision Tree']
best_single = results.loc[singles, 'Accuracy'].idxmax()
print("Best single model:", best_single, results.loc[best_single, 'Accuracy'])
print("Best voting:", results.loc[['Voting (hard)', 'Voting (soft)'], 'Accuracy'].max())
print("Best bagging:", results.loc[['Bagging (10 trees)', 'Bagging (100 trees)', 'Pasting (100 trees)'], 'Accuracy'].max())