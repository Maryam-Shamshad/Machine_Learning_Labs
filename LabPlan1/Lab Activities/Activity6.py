# Maryam Shamshad 
# 23-NTU-CS-1047
# Activity 6
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

# Explore the Dataset
data = load_breast_cancer()
print("Dataset Exploration")
print("data.keys()          :", list(data.keys()))
print("data.target_names    :", data.target_names)
print("Number of samples    :", data.data.shape[0])
print("Number of features   :", data.data.shape[1])
print("data.feature_names   :\n", data.feature_names[:5], "... (truncated for brevity)")
print("Shape of data.data   :", data.data.shape)
print("Shape of data.target :", data.target.shape)

# Conversion of Datasets to DataFrames
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

# Handle Missing Values
print("\nMissing Value Handling")
print("Missing values in raw df:", int(df.isnull().sum().sum()))
rng = np.random.default_rng(42)
missing_idx = rng.choice(df.index, size=5, replace=False)
df.loc[missing_idx, "mean radius"] = np.nan
print("Missing values after artificial injection:\n", df.isnull().sum()[df.isnull().sum() > 0])
df["mean radius"] = df["mean radius"].fillna(df["mean radius"].median())
print("Missing values left after median imputation:", int(df.isnull().sum().sum()))

# Check Class Distribution
print("\nClass Distribution & Imbalance Correction")
counts_before = df["target"].value_counts().sort_index()
print("Class counts BEFORE weighting:")
print(f" - Malignant (0): {counts_before[0]} | Benign (1): {counts_before[1]}")

# Class Weighting
sample_weights = compute_sample_weight("balanced", df["target"])
counts_after = pd.Series(sample_weights).groupby(df["target"]).sum()
print("Effective class weights AFTER balancing (sum of sample weights):")
print(f" - Malignant (0): {counts_after[0]:.1f} | Benign (1): {counts_after[1]:.1f}")

# Outlier Detection & Visualiztion
print("\n--- TASK 4: Outlier Detection via IQR ---")
feats = ["mean radius", "mean texture"]
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
for i, f in enumerate(feats):
    q1, q3 = df[f].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = df[(df[f] < low) | (df[f] > high)]
    print(f"Feature '{f}': Found {len(outliers)} outliers outside boundaries [{low:.2f}, {high:.2f}]")

    axes[i].boxplot(df[f], patch_artist=True)
    axes[i].set_title(f"Boxplot: {f}")
plt.tight_layout()
plt.show()
# We use IQR to set upper and lower boundaries in our data.
# A point that falls outside the boundary is consider as an outlier.
# On a boxplot, the main box contain the normal data, and outliers shows as dot outside the data.


# Log Transformation 
df["mean area (log)"] = np.log1p(df["mean area"])
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
axes[0].hist(df["mean area"], bins=30, color="tab:orange", edgecolor="white")
axes[0].set_title(f"Before: mean area (Skew: {df['mean area'].skew():.2f})")
axes[1].hist(df["mean area (log)"], bins=30, color="tab:blue", edgecolor="white")
axes[1].set_title(f"After: log(mean area) (Skew: {df['mean area (log)'].skew():.2f})")
for ax in axes:
    ax.set_ylabel("Frequency")
plt.tight_layout()
plt.show()
# Right-skewed data means that some numbers are creating a long tail on the right side.
# Log-transforming crash these massive numbers and pulls thae long tail inward.
# It makes the data look like a balanced and  symmetric bell curve on a histogram.

# Feature Scaling
X = df[list(data.feature_names)]
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Distance-based models calculate the physical distance between points.
# If one feature is in the thousands and another is a small decimal, the big one wins.
# Scaling normalizes brings everything to the same scale so that every feature gets equal vote.

# Reduce Scaled Features
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_train_scaled)
plt.figure(figsize=(7, 5))
for cls, colour, name in [(0, "tab:red", "malignant"), (1, "tab:green", "benign")]:
    mask = (y_train.values == cls)
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colour, alpha=0.6, edgecolor="k", linewidth=0.3, label=name)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA (2 Components) Class Separation")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
# PCA condensing 30 features into just 2 main components so we can easily look them.
# If two groups form separate clusters on a scatter plot, the classes are highly separable.

# Training two Classifiers
svm = SVC(kernel="rbf", class_weight="balanced", random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
svm.fit(X_train_scaled, y_train)
knn.fit(X_train_scaled, y_train)
svm_acc = svm.score(X_test_scaled, y_test)
knn_acc = knn.score(X_test_scaled, y_test)
print("\n--- TASK 8: Classifier Accuracy Comparison ---")
print(f"SVM (RBF Kernel) Test Accuracy : {svm_acc:.4f}")
print(f"KNN (K=5) Test Accuracy        : {knn_acc:.4f}")
# SVM tries to draw the widest possible highway line between classes.
# KNN looks at the closest points around it to make a guess.
# SVM generally performs better because boundary lines generalize better than dense distance checks.

# Preprocessing Step with the Biggest Impact
# Feature scaling had the biggest impact on dataset.
# Without scaling, the huge numerical differences completely ignore vital features.
# Standardizing the scales allows SVM and KNN to work correctly.