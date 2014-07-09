import numpy as np
from sklearn import datasets, svm, cross_validation

# Load the data set
digits = datasets.load_digits()

print 'Total instances: %s\nPossible classes: %s' % (len(digits.data), len(digits.target_names))

# Define a classifier
clf = svm.SVC(gamma=0.001, C=100)

# 10-fold cross validation
k_fold = cross_validation.StratifiedKFold(digits.target, n_folds=10)

# Get the results
result = cross_validation.cross_val_score(clf, digits.data, digits.target, cv=k_fold, n_jobs=-1)

print '\nAverage classification result: %s ' % np.mean(result)