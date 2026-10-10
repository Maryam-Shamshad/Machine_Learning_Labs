import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

wine_data = load_wine()
X_feature , Y_Target = wine_data.data , wine_data.target

#Trainning 
X_train, X_test , Y_train , Y_test = train_test_split(X_feature,Y_Target,test_size=0.2)

# Apply KNN for different values of k
k_values = range(1,30)
accuracy_scores = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)
    Y_Pred = knn.predict(X_test)
    accuracy = accuracy_score(Y_test,Y_Pred)
    accuracy_scores.append(accuracy)

# Print the best value of k
best_k = k_values[accuracy_scores.index(max(accuracy_scores))]
print("----------------------")
print("Best Value of k: ",best_k)