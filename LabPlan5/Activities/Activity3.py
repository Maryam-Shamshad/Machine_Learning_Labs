import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

# Load and display the data
df = pd.read_csv('Mall_Customers.csv')
print(df.head())

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
def plot_clusters(data, labels, title):
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=labels, cmap='viridis')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.title(title)
    plt.show()

# K-Means and DBSCAN on the original data
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
km_labels = kmeans.fit_predict(X)
plot_clusters(X, km_labels, 'K-Means (not scaled)')
dbscan = DBSCAN(eps=5, min_samples=5)
db_labels = dbscan.fit_predict(X)
plot_clusters(X, db_labels, 'DBSCAN (not scaled)  [-1 = noise]')

# Scale the features and re-run K-Means
X_scaled = StandardScaler().fit_transform(X)
km_scaled_labels = KMeans(n_clusters=5, random_state=42, n_init=10).fit_predict(X_scaled)
plot_clusters(X, km_scaled_labels, 'K-Means (scaled)')
print(pd.crosstab(km_labels, km_scaled_labels,
                  rownames=['Not scaled'], colnames=['Scaled']))

# Elbow Method to find the best k
inertia = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertia.append(km.inertia_)
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# Interpret each cluster (mean values of the K-Means clusters)
df['Cluster'] = km_labels
print(df.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean().round(1))

# Add a fake feature "LuckyNumber" (random 0-100)
np.random.seed(42)
df['LuckyNumber'] = np.random.randint(0, 101, len(df))
X_lucky = df[['Annual Income (k$)', 'Spending Score (1-100)', 'LuckyNumber']]
lucky_labels = KMeans(n_clusters=5, random_state=42, n_init=10).fit_predict(X_lucky)

# Plot with the same axes as before to see the effect
plot_clusters(X, lucky_labels, 'K-Means with LuckyNumber added')
print(pd.crosstab(km_labels, lucky_labels, rownames=['Without Lucky'], colnames=['With Lucky']))