import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load images
folder = 'brain_tumor_dataset'
images, labels = [], []
for label, name in enumerate(['no', 'yes']):          
    path = os.path.join(folder, name)
    for file in os.listdir(path):
        img = Image.open(os.path.join(path, file)).convert('L')   
        img = img.resize((64, 64))                                
        images.append(np.array(img))
        labels.append(label)

images = np.array(images)                 
X = images.reshape(len(images), -1)     
y = np.array(labels)
print("Data shape:", X.shape)

# Spliting Data
X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(
    X, y, images, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SVM with RBF kernel
svm = SVC(kernel='rbf', C=10, gamma='scale')
svm.fit(X_train, y_train)

# Evaluate
y_pred = svm.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['No Tumor', 'Tumor']))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Show a few test images with predicted and actual labels
names = ['No Tumor', 'Tumor']
plt.figure(figsize=(12, 3))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(img_test[i], cmap='gray')
    plt.title(f"Actual: {names[y_test[i]]}\nPred: {names[y_pred[i]]}", fontsize=9)
    plt.axis('off')
plt.show()