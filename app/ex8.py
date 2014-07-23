import matplotlib.pyplot as plt
import numpy as np

from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

# Generate unbalanced 2D set
np.random.seed(0)
X = np.vstack([np.random.normal(0, 1, (950, 2)),
               np.random.normal(-1.8, 0.8, (50, 2))])
y = np.hstack([np.zeros(950), np.ones(50)])

# Instantiate SVM classifier
clf = SVC()

# Splitting the data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Fit the model with training data
clf.fit(X_train, y_train)

# Get the predicted labels of test data
y_pred = clf.predict(X_test)

print classification_report(y_test, y_pred, target_names=['background', 'foreground'])

plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='none', cmap=plt.cm.Accent)
plt.show()