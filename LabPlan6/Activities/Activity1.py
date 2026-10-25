import os
import numpy as np
import pandas as pd
import kagglehub
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

path = kagglehub.dataset_download("zalando-research/fashionmnist")
print("Path to dataset files:", path)
print(os.listdir(path))     
fashion = pd.read_csv(os.path.join(path, "fashion-mnist_train.csv"))
print("Shape:", fashion.shape)
print("\nFirst 10 rows:")
print(fashion.head(10))
print("\nClass distribution:")
print(fashion["label"].value_counts().sort_index())
X = fashion.drop(columns=["label"])
y = fashion["label"]

# Normalize pixel values
X = X / 255.0

# Standardize before PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Number of features:", X.shape[1])
print("\nBefore scaling (5 example pixels):")
print(X.iloc[:, 200:205].describe().loc[["mean", "std"]])
print("\nAfter scaling (5 example pixels):")
print(pd.DataFrame(X_scaled[:, 200:205]).describe().loc[["mean", "std"]])

# Fixed Number of Components
pca_50 = PCA(n_components=50)
X_pca_50 = pca_50.fit_transform(X_scaled)
print("Method 1: n_components = 50")
print("Original number of features:", X.shape[1])
print("Retained components:", pca_50.n_components_)
print("Cumulative explained variance:", np.sum(pca_50.explained_variance_ratio_))

# Percentage of Variance
pca_95 = PCA(n_components=0.95)
X_pca_95 = pca_95.fit_transform(X_scaled)
print("Method 2: n_components = 0.95")
print("Original number of features:", X.shape[1])
print("Retained components:", pca_95.n_components_)
print("Cumulative explained variance:", np.sum(pca_95.explained_variance_ratio_))
pca_all = PCA(n_components=None)
X_pca_all = pca_all.fit_transform(X_scaled)
print("Method 3: n_components = None")
print("Original number of features:", X.shape[1])
print("Retained components:", pca_all.n_components_)
print("Cumulative explained variance:", np.sum(pca_all.explained_variance_ratio_))
cumsum_all = np.cumsum(pca_all.explained_variance_ratio_)
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumsum_all) + 1), cumsum_all)
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance (Fashion MNIST)')
plt.grid()
plt.show()
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca_all[:, 0], X_pca_all[:, 1], c=y, cmap='jet', edgecolors='none', alpha=0.6)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA Visualization of Fashion MNIST')
cbar = plt.colorbar(scatter, ticks=range(10), label='Clothing Class')
cbar.ax.set_yticklabels(class_names)
plt.grid(alpha=0.2)
plt.show()
comparison = pd.DataFrame({
    "Method": ["1. Fixed (n_components=50)",
               "2. Variance (n_components=0.95)",
               "3. All (n_components=None)"],
    "Original Features": [X.shape[1]] * 3,
    "Retained Components": [pca_50.n_components_,
                            pca_95.n_components_,
                            pca_all.n_components_],
    "Cumulative Explained Variance": [np.sum(pca_50.explained_variance_ratio_),
                                      np.sum(pca_95.explained_variance_ratio_),
                                      np.sum(pca_all.explained_variance_ratio_)]
})
print(comparison.to_string(index=False))
most_reduction = comparison.loc[comparison["Retained Components"].idxmin(), "Method"]
most_info = comparison.loc[comparison["Cumulative Explained Variance"].idxmax(), "Method"]
print("\nGreatest dimensionality reduction:", most_reduction)
print("Preserves the most information:", most_info)