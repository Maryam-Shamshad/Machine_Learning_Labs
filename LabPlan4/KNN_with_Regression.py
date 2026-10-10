import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

wine_data = load_wine(as_frame=True)
df = wine_data.frame

# Continuous target: alcohol. Features: everything except alcohol and the class label
X_feature = df.drop(columns=["alcohol", "target"])
Y_Target = df["alcohol"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X_feature, Y_Target, test_size=0.2, random_state=42
)

k_values = [1, 3, 5, 7, 9]
rmse_scores = []
r2_scores = []

for k in k_values:
    # Scaling matters for KNN because it is distance-based
    knn = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=k))
    knn.fit(X_train, Y_train)
    preds = knn.predict(X_test)

    rmse_scores.append(np.sqrt(mean_squared_error(Y_test, preds)))
    r2_scores.append(r2_score(Y_test, preds))

# Plot error and R² against k
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].plot(k_values, rmse_scores, marker="o")
ax[0].set_xlabel("k")
ax[0].set_ylabel("RMSE")
ax[0].set_title("RMSE vs. k")

ax[1].plot(k_values, r2_scores, marker="o", color="green")
ax[1].set_xlabel("k")
ax[1].set_ylabel("R² score")
ax[1].set_title("R² vs. k")

plt.tight_layout()
plt.show()