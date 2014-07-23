import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

digits = load_digits()
X, y = digits.data, digits.target

C_range = np.logspace(-2, 2, 40)
clf = SVC()

# It would be better to use stratified K-fold
grid = GridSearchCV(clf, param_grid={'C': C_range}, scoring='f1', cv=10, n_jobs=-1)
grid.fit(X, y)

print 'Best C found: {0}'.format(grid.best_params_)

scores = [g[1] for g in grid.grid_scores_]

plt.semilogx(C_range, scores)
plt.xlabel('Parameter C')
plt.ylabel('F1 score')
plt.grid()

plt.show()