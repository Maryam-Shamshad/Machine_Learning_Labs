# Maryam Shamshad 
# 23-NTU-CS-1047
# Exploring Dataset using fetch oliveti faces
from sklearn.datasets import fetch_olivetti_faces
faces = fetch_olivetti_faces()
faces.keys()
n_samples, n_features = faces.data.shape
print((n_samples, n_features))
print(faces.images.shape)
print(faces.data.shape)
print(faces.DESCR)
X,y=faces.data,faces.target
import matplotlib.pyplot as plt
plt.imshow(faces.images[5], cmap="gray")
plt.title(f"Face ID: {faces.target[5]}")
plt.axis("off")
plt.show()
plt.figure(figsize=(6, 6))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(faces.images[i], cmap="gray")
    plt.axis("off")
    plt.title(faces.target[i])
plt.show()