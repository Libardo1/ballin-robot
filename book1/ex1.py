from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

from collections import Counter


def print_scatter(x_train, x_train_scaled):
    # Scatter two versions
    fig, axes = plt.subplots(nrows=1, ncols=2)

    for i in xrange(len(colors)):
        xs = x_train[:, 0][y_train == i]
        ys = x_train[:, 1][y_train == i]

        xss = x_train_scaled[:, 0][y_train == i]
        yss = x_train_scaled[:, 1][y_train == i]

        axes[0].scatter(xs, ys, c=colors[i])
        axes[1].scatter(xss, yss, c=colors[i])

    for ax in axes:
        ax.set_xlabel('Sepal length')
        ax.set_ylabel('Sepal width')

    axes[0].set_title('Without feature scaling')
    axes[1].set_title('With feature scaling')
    plt.show()

iris = load_iris()
X, y = iris.data, iris.target

# Take only two first features
X = X[:, :2]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

train_labels = Counter(y_train)
test_labels = Counter(y_test)

print 'Train dataset distribution:'
for k, v in train_labels.iteritems():
    print '\tClass {0} elements: {1:.2f}%'.format(k, v / float(len(y_train)) * 100)

print 'Test dataset distribution:'
for k, v in test_labels.iteritems():
    print '\tClass {0} elements: {1:.2f}%'.format(k, v / float(len(y_test)) * 100)

# Stochastic Gradient Descend classifier
clf = SGDClassifier()
colors = ['red', 'greenyellow', 'blue']

# SGD without scaling
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print '\nResults for SGD without feature scaling'
print classification_report(y_test, y_pred, target_names=iris.target_names)

# SGD with feature scaling
scaler = StandardScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf.fit(X_train_scaled, y_train)
y_pred_scaled = clf.predict(X_test_scaled)
print '\nResults for SGD with feature scaling'
print classification_report(y_test, y_pred_scaled, target_names=iris.target_names)

# print_scatter(X_train, X_train_scaled)