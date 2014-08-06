import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.cross_validation import train_test_split, KFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC

from scipy.stats import sem
import numpy as np

faces = fetch_olivetti_faces()
X_train, X_test, y_train, y_test = train_test_split(faces.data, faces.target, test_size=0.25, random_state=0)


def print_faces(images, target, top_n):
    fig = plt.figure(figsize=(12, 12))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in xrange(top_n):
        p = fig.add_subplot(20, 20, i + 1, xticks=[], yticks=[])
        p.imshow(images[i], cmap=plt.cm.bone)

        p.text(0, 14, str(target[i]))
        p.text(0, 60, str(i))

    plt.show()


def evaluate_cross_validation(clf, X, y, K):
    cv = KFold(len(y), K, shuffle=True, random_state=0)
    scores = cross_val_score(clf, X, y, cv=cv)

    print scores
    print "Mean score: {0:.3f} (+/-{1:.3f})".format(scores.mean(), sem(scores))


def train_and_evaluate(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)

    print 'Accuracy on training set: {0:.3f}'.format(clf.score(X_train, y_train))
    print 'Accuracy on testing set : {0:.3f}'.format(clf.score(X_test, y_test))

    y_pred = clf.predict(X_test)

    print 'Classification report:'
    print classification_report(y_test, y_pred)

svc_1 = SVC(kernel='linear')
train_and_evaluate(svc_1, X_train, X_test, y_train, y_test)