# Maryam Shamshad 
# 23-NTU-CS-1047
# Data Preporcessing
# Handling Imbalance Data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
np.random.seed(42)
X = pd.DataFrame({
    'Feature1': np.random.rand(1000),
    'Feature2': np.random.rand(1000)
})
y = pd.Series([0]*900 + [1]*100)
print("Original class distribution:\n", y.value_counts())
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Oversampling (SMOTE)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print("\nAfter SMOTE (oversampling):\n", y_train_smote.value_counts())

# Undersampling
undersample = RandomUnderSampler(random_state=42)
X_train_under, y_train_under = undersample.fit_resample(X_train, y_train)
print("\nAfter Undersampling:\n", y_train_under.value_counts())

# Class Weighting
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train) # trained on the original imbalanced data
print("\nClass weighting model trained on original (unbalanced) data")
print("Model automatically weights the minority class more heavily during training")