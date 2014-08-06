import numpy as np

from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.metrics import classification_report

from scipy.stats import sem

SPLIT_PERC = 0.75

news = fetch_20newsgroups(subset='all')

split_size = int(len(news.data)*SPLIT_PERC)

X_train = news.data[:split_size]
X_test = news.data[split_size:]

y_train = news.target[:split_size]
y_test = news.target[split_size:]

clf = Pipeline([
    ('vect', TfidfVectorizer(stop_words='english')),
    ('clf', MultinomialNB(alpha=0.1))
])

clf_1 = Pipeline([
    ('vect', TfidfVectorizer(stop_words='english')),
    ('clf', SVC())
])


def evaluate_cross_validation(clf, X, y, K):
    cv = KFold(len(y), K, shuffle=True, random_state=0)
    scores = cross_val_score(clf, X, y, cv=cv)
    print scores
    print 'Mean score: {0:.3f} (+/- {1:.3f})'.format(np.mean(scores), sem(scores))


def train_and_evaluate(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)

    print 'Accuracy on training set: {0:.3f}'.format(clf.score(X_train, y_train))
    print 'Accuracy on testing set: {0:.3f}'.format(clf.score(X_test, y_test))

    y_pred = clf.predict(X_test)
    print 'Classification report'
    print classification_report(y_test, y_pred)


train_and_evaluate(clf_1, X_train, X_test, y_train, y_test)