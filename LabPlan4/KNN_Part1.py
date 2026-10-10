import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

wine_data = load_wine()
X_feature , Y_Target = wine_data.data , wine_data.target

#Trainning 
X_train, X_test , Y_train , Y_test = train_test_split(X_feature,Y_Target,test_size=0.2)

#Requirment for K values of model fit 
k_values = [1,3,5,7,9]
accuracy_scores = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)
    accuracy = knn.score(X_test,Y_test)
    accuracy_scores.append(accuracy)
    
# plot accuracy against k values
plt.scatter(k_values,accuracy_scores)
plt.xlabel("k")
plt.ylabel("Accuracy vs. k values")
plt.show()