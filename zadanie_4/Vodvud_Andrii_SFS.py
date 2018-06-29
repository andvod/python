
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.base import clone
from sklearn.base import BaseEstimator
from sklearn.base import MetaEstimatorMixin
from itertools import combinations
from sklearn.cross_validation import cross_val_score
import numpy as np
import sys

class SFS(BaseEstimator, MetaEstimatorMixin):
    """ Sequential Forward Selection for feature selection.

    Parameters
    ----------
    estimator : scikit-learn estimator object

    print_progress : bool (default: True)
       Prints progress as the number of epochs
       to stderr.

    k_features : int
      Number of features to select where k_features.

    scoring : str, (default='accuracy')
      Scoring metric for the cross validation scorer.

    cv : int (default: 5)
      Number of folds in StratifiedKFold.

    n_jobs : int (default: 1)
      The number of CPUs to use for cross validation. -1 means 'all CPUs'.

    Attributes
    ----------
    indices_ : array-like, shape = [n_predictions]
      Indices of the selected subsets.

    k_score_ : float
      Cross validation mean score of the selected subset

    subsets_ : list of tuples
      Indices of the sequentially selected subsets.

    scores_ : list
      Cross validation mean scores of the sequentially selected subsets.

    Examples
    --------
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> X = iris.data
    >>> y = iris.target
    >>> knn = KNeighborsClassifier(n_neighbors=4)
    >>> sfs = SFS(knn, k_features=2, scoring='accuracy', cv=5)
    >>> sfs = sfs.fit(X, y)
    >>> sfs.indices_
    (2, 3)
    >>> sfs.transform(X[:5])
    array([[ 1.4,  0.2],
           [ 1.4,  0.2],
           [ 1.3,  0.2],
           [ 1.5,  0.2],
           [ 1.4,  0.2]])

    >>> print('best score: %.2f' % sfs.k_score_)
    best score: 0.97

    """
    def __init__(self, estimator, k_features, print_progress=True,
                 scoring='accuracy', cv=5, n_jobs=1):
        self.scoring = scoring
        self.estimator = estimator #clone(estimator)
        self.cv = cv
        self.k_features = k_features
        self.print_progress = print_progress
        self.n_jobs = n_jobs

    def fit(self, X, y):

        dim = 0
        orig_set = set(range(X.shape[1]))
        self.indices_ = []
        self.subsets_ = []
        self.scores_ = []

        while dim < self.k_features:
            scores = []
            subsets = []

            set_indices = set(self.indices_)
            for i in orig_set - set_indices:
                test_subset = tuple(sorted(set_indices | {i}))
                cv_score = self._calc_score(X, y, test_subset)
                scores.append(cv_score.mean())
                subsets.append(test_subset)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            self.scores_.append(scores[best])
            dim += 1

            if self.print_progress:
                sys.stderr.write('\rFeatures: %d/%d' % (len(self.indices_), self.k_features))
                sys.stderr.flush()


        self.k_score_ = self.scores_[-1]
        return self

    def transform(self, X):
        return X[:, self.indices_]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def _calc_score(self, X, y, indices):
        cv_score = cross_val_score(self.estimator,
                                   X[:, indices], y,
                                   cv=self.cv,
                                   scoring=self.scoring,
                                   n_jobs=self.n_jobs)
        return cv_score


breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

knn = KNeighborsClassifier(n_neighbors=7)

sfs = SFS(knn, k_features=7, scoring='accuracy', cv=7)
sfs.fit(X, y)

print('Indeksy wybranych cech:', sfs.indices_)
print('Sprawnosc klasyfikatora dla wybranego podzbioru cech:',
      sfs.k_score_)

