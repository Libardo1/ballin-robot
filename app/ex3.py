from sklearn import tree, datasets

# Visualizing decision tree

iris = datasets.load_iris()

clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

# Create PDF with command:
# dot -Tpdf iris.dot -o iris.pdf
with open("iris.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f, feature_names=iris.feature_names)