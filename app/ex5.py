import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA

iris = datasets.load_iris()
X, y = iris.data, iris.target

# Principle Component Analysis
# Reduce a number of features
pca = PCA(n_components=2)
pca.fit(X)
X_reduced = pca.transform(X)

print "Reduced data set shape: {0}".format(X_reduced.shape)
print "Meaning of 2 components:"

for i, component in enumerate(pca.components_):
    print "[{:d}]".format(i)
    print " + ".join("%.3f x %s" % (value, name) for value, name in zip(component, iris.feature_names))

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y)
plt.show()

