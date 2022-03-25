from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from .base import BaseClassifier


class BestKNeighborsClassifier(BaseClassifier):
    estimator = KNeighborsClassifier
    param_grid = {
        'n_neighbors': range(1, 21, 2),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }


class BestSVC(BaseClassifier):

    estimator = SVC
    param_grid = {
        'probability': [True],
        'class_weight': ['balanced'],
        'C': [0.1, 1, 10, 100], 
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }

class BestRidgeClassifier(BaseClassifier):

    estimator = RidgeClassifier
    param_grid = {
        'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    }


class BestRandomForestClassifier(BaseClassifier):
    estimator = RandomForestClassifier
    param_grid = {
        'n_estimators': [50, 100, 250],
        'max_depth': [5, 10, 20]
    }

class BestLogisticRegression(BaseClassifier):
    estimator = LogisticRegression
    param_grid = {
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'penalty': ['none', 'l1', 'l2', 'elasticnet'],
        'C': [100, 10, 1.0, 0.1, 0.01]
    }

