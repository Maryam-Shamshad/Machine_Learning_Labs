# Maryam Shamshad
# 23-NTU-CS-1047
# Activity 1: Logistic Regression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# Loading data
data = pd.read_csv("Social_Network_Ads.csv")
features = ['Gender', 'Age', 'EstimatedSalary', 'Purchased']
data = data[features].dropna()

# Encoding and Spliting Datset 
le = LabelEncoder()
for col in ['Gender']:
    data[col] = le.fit_transform(data[col])
X = data.drop('Purchased', axis=1)
y = data['Purchased']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print("Classification Report:\n", classification_report(y_test, y_pred))

# Plot Confusion Matrix and ROC Curve
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['Not Purchased', 'Purchased'], cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='darkorange', label=f'ROC Curve (AUC = {auc(fpr, tpr):.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()