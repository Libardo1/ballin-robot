import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from sklearn import metrics

# Confusion matrix

digits = load_digits()
X, y = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y)

clf = GaussianNB()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print 'Classification accuracy: {0:.02f}%'.format(metrics.accuracy_score(y_test, y_pred))

plt.imshow(metrics.confusion_matrix(y_test, y_pred), cmap=plt.cm.binary, interpolation='nearest')
plt.colorbar()

plt.title('Confusion matrix')
plt.xlabel('True value')
plt.ylabel('Predicted value')

plt.show()