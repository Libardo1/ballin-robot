import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# PCA + Clustering example

iris = datasets.load_iris()
X = iris.data

pca = PCA(n_components=2)
pca.fit(X)
X_reduced = pca.transform(X)

k_means = KMeans(n_clusters=3, random_state=0)
k_means.fit(X)

y_pred = k_means.predict(X)

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_pred)
plt.show()