import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

wine = load_wine()

x = wine.data
y = wine.target
feat_names = wine.feature_names
target_names = wine.target_names

print("Original dataset Shape: ", x.shape)
print("Target shape: ", y.shape)
print("Feature names: ", feat_names)
print("Target names: ",target_names)

# part 2
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)

print("Original dataset Shape: ", x.shape)
print("Reduced dataset Shape: ", X_reduced.shape)

wine_colors = ['orangered', 'darkred', 'crimson']

plt.figure(figsize=(10,6))

for class_num, color, target_names in zip(
    [0, 1, 2], wine_colors, target_names
):
    plt.scatter(
        X_reduced[y == class_num, 0],
        X_reduced[y == class_num, 1],
        color=color,
        label=target_names,
        alpha=0.7
    )
plt.title("PCA for Wine Dataset")
plt.xlabel("First Principal Componenet")
plt.ylabel("Second Principal Componenet")
plt.legend()
plt.grid(alpha=0.2)
plt.show()

print("Principal Component Loading")
print(pca.components_)

print("Shape of Principal Components: ", pca.components_.shape)

plt.figure(figsize=(14,5))

sns.heatmap(
    pca.components_,
    cmap='viridis',
    xticklabels=feat_names,
    yticklabels=["First Components", "Second Component"]
)

plt.title("PCA for Wine Loading")
plt.xlabel("Original Features")
plt.ylabel("Principal Componenets")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print("Explained variance:")
print(pca.explained_variance_)

print("\nExplained varaince ratio:")
print(pca.explained_variance_ratio_)


pca_3 = PCA(n_components=0.96)
X_reduced_96=pca_3.fit_transform(X_scaled)

print("Number of componenets selected:", pca_3.n_components_)

print("Total Explained Variance:",
      np.sum(pca_3.explained_variance_ratio_))
print("Reduced Dataset shape:",X_reduced_96.shape)