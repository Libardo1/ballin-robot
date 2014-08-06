import csv
import numpy as np
import pydot, StringIO

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split, LeaveOneOut
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
from scipy.stats import sem

with open('../data/titanic.csv', 'rb') as csvfile:
    titanic_reader = csv.reader(csvfile, delimiter=',', quotechar='"')

    # Header contains feature names
    row = titanic_reader.next()
    feature_names = np.array(row)

    # Load dataset and target classes
    titanic_X, titanic_y = [], []

    for row in titanic_reader:
        titanic_X.append(row)
        titanic_y.append(row[2])  # The target value is 'survived'

    titanic_X = np.array(titanic_X)
    titanic_y = np.array(titanic_y)

## Preprocessing the data
# We keep class, age and sex

titanic_X = titanic_X[:, [1, 4, 10]]
feature_names = feature_names[[1, 4, 10]]

# Dealing with missing features
ages = titanic_X[:, 1]
mean_age = np.mean(titanic_X[ages != 'NA', 1].astype(np.float))
titanic_X[titanic_X[:, 1] == 'NA', 1] = mean_age

# Encode sex label
enc = LabelEncoder()
sex_encoder = enc.fit(titanic_X[:, 2])
t = sex_encoder.transform(titanic_X[:, 2])
titanic_X[:, 2] = t

# Encode class label
class_encoder = enc.fit(titanic_X[:, 0])
integer_classes = class_encoder.transform(class_encoder.classes_).reshape(3, 1)

enc = OneHotEncoder()
one_hot_encoder = enc.fit(integer_classes)
num_of_rows = titanic_X.shape[0]

t = class_encoder.transform(titanic_X[:, 0]).reshape(num_of_rows, 1)

new_features = one_hot_encoder.transform(t)

titanic_X = np.concatenate([titanic_X, new_features.toarray()], axis=1)
titanic_X = np.delete(titanic_X, [0], 1)

feature_names = ['age', 'sex', '1st class', '2nd class', '3rd class']

titanic_X = titanic_X.astype(float)
titanic_y = titanic_y.astype(float)

## Classification
X_train, X_test, y_train, y_test = train_test_split(titanic_X, titanic_y, test_size=0.25, random_state=33)
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=5)

clf = clf.fit(X_train, y_train)

dot_data = StringIO.StringIO()
export_graphviz(clf, out_file=dot_data, feature_names=feature_names)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_png("titanic.png")

def measure_performance(X, y, clf, show_accuracy=True,
                        show_classification_report=False, show_confusion_matrix=False):
    y_pred = clf.predict(X)

    if show_accuracy:
        print "Accuracy: {0:.3f}\n".format(metrics.accuracy_score(y, y_pred))

    if show_classification_report:
        print 'Classification report'
        print metrics.classification_report(y, y_pred)

    if show_confusion_matrix:
        print 'Confusion matrix'
        print metrics.confusion_matrix(y, y_pred)


def loo_cv(X_train, y_train, clf):
    loo = LeaveOneOut(X_train[:].shape[0])
    scores = np.zeros(X_train[:].shape[0])

    for train_index, test_index in loo:
        X_train_cv, X_test_cv = X_train[train_index], X_train[test_index]
        y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]

        clf = clf.fit(X_train_cv, y_train_cv)
        y_pred = clf.predict(X_test_cv)

        scores[test_index] = metrics.accuracy_score(y_test_cv.astype(int), y_pred.astype(int))

    print "Mean score: {0:.3f} (+/-{1:.3f})".format(np.mean(scores), sem(scores))

loo_cv(X_train, y_train, clf)