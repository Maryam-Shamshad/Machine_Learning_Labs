import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import fetch_openml

# Loading the Dataset
elec_data = fetch_openml(name='electricity', version=1)
elec_df = elec_data.frame

print("Data shape:", elec_data.data.shape)
print("Feature names:", elec_data.feature_names)
print("Target names:", elec_data.target_names)
elec_df['day'] = elec_df['day'].astype(int)

# Spliting the Dataset
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(elec_df, test_size=0.25, random_state=20)

print("Training Samples: ",len(train_data))
print("Testing Samples: ",len(test_data))

# Quick Look at the Data
train_data.head(10)

# Dataset Information
train_data.info()

# Summary Statistics
train_data.describe()

# Missing Values
train_data.isnull().sum()

# Correlation HeatMap
corr = train_data.corr(numeric_only=True)
plt.figure(figsize=(12, 7))
sns.heatmap(corr, annot=True, cmap='crest')
plt.show()

# Class Distribution
plt.figure(figsize=(12, 7))
sns.countplot(data=train_data, x='class')
plt.show()

# Class by Day of Week
plt.figure(figsize=(13, 8))
sns.countplot(data=train_data, x='day', hue='class')
plt.show()

# Feature Histogram
train_data.hist(bins=50, figsize=(15, 10))
plt.show()

# Pre-processing
from sklearn.preprocessing import LabelEncoder

X_train = train_data.drop('class', axis=1)
y_train = train_data['class']

label_enc = LabelEncoder()
y_train_prepared = label_enc.fit_transform(y_train)
print("Classes:", label_enc.classes_)

# Accuracy Helper
from sklearn.metrics import accuracy_score

def accuracy(model, data, labels):
    predictions = model.predict(data)
    acc = accuracy_score(labels, predictions)
    return acc

# Voting Code
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

models = [
    ('lr', LogisticRegression(max_iter=1000)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('dt', DecisionTreeClassifier(max_depth=5, random_state=42)) 
]

voting_hard = VotingClassifier(models, voting='hard')
voting_soft = VotingClassifier(models, voting='soft')

print("Training Hard & Soft Voting Classifiers")
voting_hard.fit(X_train, y_train_prepared)
voting_soft.fit(X_train, y_train_prepared)
print("Voting Hard Train Acc:", accuracy(voting_hard, X_train, y_train_prepared))
print("Voting Soft Train Acc:", accuracy(voting_soft, X_train, y_train_prepared))

# Bagging Classifier
from sklearn.ensemble import BaggingClassifier

print("Training Bagging Classifier")
bag_classifier = BaggingClassifier(
    DecisionTreeClassifier(class_weight='balanced'),
    max_samples=0.5, max_features=0.5, bootstrap=False
)
bag_classifier.fit(X_train, y_train_prepared)
print("Bagging Train Acc:", accuracy(bag_classifier, X_train, y_train_prepared))

# Gradient Boosting Code
from sklearn.ensemble import GradientBoostingClassifier

print("Training Gradient Boosting Classifier")
grad_boost_clf = GradientBoostingClassifier(
    n_estimators=500,
    learning_rate=0.8,
    random_state=42,
    max_depth=2
)
grad_boost_clf.fit(X_train, y_train_prepared)
print("Gradient Boosting Train Acc:", accuracy(grad_boost_clf, X_train, y_train_prepared))

# Adaboost Code 
from sklearn.ensemble import AdaBoostClassifier

print("Training AdaBoost Classifier")
adaboost_clf = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=3, class_weight='balanced'),
    n_estimators=300,
    learning_rate=0.5
)
adaboost_clf.fit(X_train, y_train_prepared)
print("AdaBoost Train Acc:", accuracy(adaboost_clf, X_train, y_train_prepared))

# XGBoost Code
import xgboost as xgb

print("Training XGBoost Classifier")
xgb_clf = xgb.XGBClassifier()
xgb_clf.fit(X_train, y_train_prepared)
print("XGBoost Train Acc:", accuracy(xgb_clf, X_train, y_train_prepared))

# Stacking Code
from sklearn.ensemble import StackingClassifier

print("Training Stacking Classifier")
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('dt', DecisionTreeClassifier(max_depth=5, random_state=42)),
    ('lr', LogisticRegression(max_iter=1000)),
]

stack_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5
)
stack_clf.fit(X_train, y_train_prepared)
print("Stacking Train Acc:", accuracy(stack_clf, X_train, y_train_prepared))

# Evaluation on Test Data
print("\n Final Test Set Evaluation")
X_test = test_data.drop('class', axis=1)
y_test = test_data['class']
y_test_prepared = label_enc.transform(y_test)

print("Gradient Boosting Test Acc:", accuracy(grad_boost_clf, X_test, y_test_prepared))
print("Bagging Test Acc:", accuracy(bag_classifier, X_test, y_test_prepared))
print("Hard Voting Test Acc:", accuracy(voting_hard, X_test, y_test_prepared))
print("Soft Voting Test Acc:", accuracy(voting_soft, X_test, y_test_prepared))
print("XGBoost Test Acc:", accuracy(xgb_clf, X_test, y_test_prepared))
print("Stacking Test Acc:", accuracy(stack_clf, X_test, y_test_prepared))
print("AdaBoost Test Acc:", accuracy(adaboost_clf, X_test, y_test_prepared))