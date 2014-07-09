from sklearn import datasets, svm
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics.metrics import classification_report

# Grid Search Example

# Load the data set
digits = datasets.load_digits()

n_samples = len(digits.images)
X = digits.images.reshape(n_samples, -1)
y = digits.target

# Split the data set into two subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

param_grid = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
]

scores = ['precision', 'recall']

for score in scores:
    print '\nTuning hyper parameters for %s\n' % score

    # Define a classifier
    clf = GridSearchCV(svm.SVC(), param_grid, cv=5, n_jobs=-1, scoring=score)
    clf.fit(X_train, y_train)

    print 'Best parameters set found on development set:\n'
    print clf.best_estimator_

    print 'Grid scores on development set:\n'
    for params, mean_score, scores in clf.grid_scores_:
        print ("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() / 2, params))

    print '\nDetailed classification report\n'
    print 'The model is trained on the full development set.'
    print 'The scores are computed on the full evaluation set\n'

    y_true, y_pred = y_test, clf.predict(X_test)
    print classification_report(y_true, y_pred)