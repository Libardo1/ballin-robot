"""
Faces recognition example using eigenfaces and SVMs
"""

import logging
from time import time

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

print __doc__

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Download the data
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# Introspect the images array to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

target_names = lfw_people.target_names
n_features = lfw_people.data.shape[1]
n_classes = target_names.shape[0]

X, y = lfw_people.data, lfw_people.target

print "Total dataset size:"
print "n_samples: {0}".format(n_samples)
print "n_features: {0}".format(n_features)
print "n_classes: {0}".format(n_classes)

# It should be done with StratifiedKFold
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Calculate the PCA (eigenfaces)
n_components = 150

print "\nExtracting the top {0} eigenfaces from {1} faces".format(n_components, X_train.shape[0])
t0 = time()
pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
print "Done in {:.03f}s".format(time()-t0)

eigenfaces = pca.components_.reshape((n_components, h, w))
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]

print "\nProjecting the input data on the eigenfaces orthonormal basis"
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print "Done in {:.3f}s".format(time()-t0)

# Training the SVM classification model
print "\nFitting the classifier to the training set"
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1] }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto'), param_grid)
clf = clf.fit(X_train_pca, y_train)
print "Done in {:.3f}s".format(time()-t0)
print "Best estimator found by grid search: {0}".format(clf.best_estimator_)
print "Estimator score {0}".format(clf.best_score_)

# Evaluation of model quality
print "Predicting people's names on the test set"
t0 = time()
y_pred = clf.predict(X_test_pca)
print "Done in {:.3f}s".format(time()-t0)
print classification_report(y_test, y_pred, target_names=target_names)
print confusion_matrix(y_test, y_pred, labels=range(n_classes))


def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)

    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i) for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()