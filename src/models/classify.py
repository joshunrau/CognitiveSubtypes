from abc import abstractmethod
from typing import Callable, Type

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .base import BaseModel
from ..utils import is_instantiated


class BaseClassifier(BaseModel):

    available_metrics = {
        'accuracy': accuracy_score,
        'balanced_accuracy': balanced_accuracy_score
    }

    def __init__(self, score_method: str) -> None:
        super().__init__()
        self.score_method = score_method
        self.grid = None
        self.n_targets = None

    @property
    def estimator(self) -> ClassifierMixin | Type[ClassifierMixin]:
        return self._estimator

    @estimator.setter
    def estimator(self, est) -> None:
        if not isinstance(est, ClassifierMixin):
            raise TypeError
        check_is_fitted(est)
        self._estimator = est

    @property
    @abstractmethod
    def param_grid(self) -> dict:
        pass

    def is_fitted(self) -> bool:
        if not is_instantiated(self.estimator):
            return False
        return True

    def fit(self, X: np.array, y: np.array) -> None:
        if self.score_method not in self.available_metrics.keys():
            raise ValueError(f"Scoring must be one of: {self.available_metrics.keys()}")
        check_X_y(X=X, y=y)
        self._X, self._y = X, y
        self.grid = GridSearchCV(self.estimator(), self.param_grid, n_jobs=-1, scoring=self.score_method)
        self.grid.fit(X, y)
        self.n_targets = len(np.unique(y))
        self.estimator = self.grid.best_estimator_

    def predict(self, X: np.array) -> None:
        check_array(X)
        return self.grid.predict(X)

    def predict_proba(self, X: np.array):
        self.check_is_fitted()
        check_array(X)
        return self.estimator.predict_proba(X)

    def decision_function(self, X: np.array):
        self.check_is_fitted()
        check_array(X)
        return self.estimator.decision_function(X)

    def compute_metric(self, metric: Callable, X: np.array, y: np.array):
        return metric(self.predict(X), y)

    def score(self, X: np.array, y: np.array):
        return self.compute_metric(self.available_metrics[self.score_method], X, y)


class BestKNeighborsClassifier(BaseClassifier):
    sklearn_estimator = KNeighborsClassifier
    param_grid = {
        'n_neighbors': range(1, 21, 2),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }


class BestSVC(BaseClassifier):
    sklearn_estimator = SVC
    param_grid = {
        'probability': [True],
        'class_weight': ['balanced'],
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }


class BestRidgeClassifier(BaseClassifier):
    sklearn_estimator = RidgeClassifier
    param_grid = {
        'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    }


class BestRandomForestClassifier(BaseClassifier):
    sklearn_estimator = RandomForestClassifier
    param_grid = {
        'n_estimators': [50, 100, 250],
        'max_depth': [5, 10, 20]
    }


class BestLogisticRegression(BaseClassifier):
    sklearn_estimator = LogisticRegression
    param_grid = {
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'penalty': ['none', 'l1', 'l2', 'elasticnet'],
        'C': [100, 10, 1.0, 0.1, 0.01]
    }
