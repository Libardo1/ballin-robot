import numpy as np
import warnings
from time import time

from sklearn.datasets import load_digits
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.svm import SVC

"""
Performing grid search with stratified K-fold over digits dataset
"""

if __name__ == "__main__":
    warnings.simplefilter('ignore')

    digits = load_digits()
    X, y = digits.data, digits.target

    print "\nInitial dataset information"
    print "\tNumber of instances: {0}".format(X.shape[0])
    print "\tNumber of features: {0}".format(X.shape[1])
    print "\tNumber of classes: {0}".format(len(digits.target_names))

    estimators = [
        ('reduce_dim', PCA()),
        ('svm', SVC())
    ]

    params = {
        'reduce_dim__n_components': np.logspace(3, 5, base=2, num=5).astype('int'),
        'svm__C': [0.1, 10, 100],
        'svm__kernel': ['linear', 'rbf']
    }

    scores = ['precision', 'recall', 'f1']

    clf = Pipeline(estimators)

    for score in scores:
        grid_search = GridSearchCV(clf, params, cv=StratifiedKFold(y, 10, shuffle=True), n_jobs=-1, scoring=score, verbose=1)

        print '\nLearning classifier for score \'{0}\''.format(score)
        t0 = time()
        grid_search.fit(X, y)
        print 'Done in {:.3f}s'.format(time() - t0)

        print '\nResults'
        print '\tBest {0} score: {1}'.format(score, grid_search.best_score_)
        print '\tBest parameters: {0}'.format(grid_search.best_params_)
