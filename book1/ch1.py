from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report

from collections import Counter

iris = load_iris()
X, y = iris.data, iris.target

# Take only two first features
X = X[:,:2]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

train_labels = Counter(y_train)
test_labels = Counter(y_test)

print 'Train dataset distribution:'
for k,v in train_labels.iteritems():
    print '\tClass {0} elements: {1:.2f}%'.format(k, v/float(len(y_train))*100)

print 'Test dataset distribution:'
for k,v in test_labels.iteritems():
    print '\tClass {0} elements: {1:.2f}%'.format(k, v/float(len(y_test))*100)

# Stochastic Gradient Descend classifier
clf = SGDClassifier()

# SGD without scaling
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print '\nResults for SGD without feature scaling'
print classification_report(y_test, y_pred, target_names=iris.target_names)

# SGD with feature scalling
scaler = StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print '\nResults for SGD with feature scalling'
print classification_report(y_test, y_pred, target_names=iris.target_names)
