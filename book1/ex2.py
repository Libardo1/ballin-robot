from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import sem

iris = load_iris()
X, y = iris.data, iris.target

clf = Pipeline([
    ('scaler', StandardScaler()),
    ('linear_model', SGDClassifier())
])

cv = KFold(X.shape[0], 5, shuffle=True)
scores = cross_val_score(clf, X, y, cv=cv)

print scores.mean(), sem(scores)
