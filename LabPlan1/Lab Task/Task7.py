# Maryam Shamshad 
# 23-NTU-CS-1047
# Loading Dataset
# Using Sklearn___Loading Data from Iris Dataset, Showing all Features and Target Values
from sklearn import datasets
iris = datasets.load_iris()
print("Features (first 5 rows):")
print(iris.data[:5])
print("\nTarget values (first 5 rows):")
print(iris.target[:5])