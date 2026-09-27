# Maryam Shamshad 
# 23-NTU-CS-1047
# # Exploring Dataset using iris Dataset
from sklearn.datasets import load_iris
iris = load_iris()
print(iris.keys())
print(iris["target_names"])
n_samples, n_features = iris.data.shape
print('Number of samples:', n_samples)
print('Number of features:', n_features)
print('Feature Name:', iris.feature_names)
print('Target Name:', iris.target_names)
print("Dimension of Input", iris.data.shape)
print("Dimension of Output", iris.target.shape)
print("First 5 rows of Input:", iris.data[1:5])
print("Target Data:", iris.target)