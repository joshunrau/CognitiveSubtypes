from abc import abstractmethod
from typing import Callable, Type

import numpy as np

from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from skopt import BayesSearchCV

from .base import BaseModel
from ..utils import is_instantiated


class BaseClassifier(BaseModel):

    available_metrics = {
        'accuracy': accuracy_score,
        'balanced_accuracy': balanced_accuracy_score
    }

    def __init__(self, score_method: str = 'accuracy') -> None:
        super().__init__()
        self.score_method = score_method
        self.grid_ = None
        self.n_targets_ = None

    @property
    def subestimator(self) -> ClassifierMixin | Type[ClassifierMixin]:
        return self._subestimator

    @subestimator.setter
    def subestimator(self, est) -> None:
        if not isinstance(est, ClassifierMixin):
            raise TypeError
        check_is_fitted(est)
        self._subestimator = est

    @property
    @abstractmethod
    def param_grid(self) -> dict:
        pass

    @property
    @abstractmethod
    def n_iter(self) -> int:
        pass

    def is_fitted(self) -> bool:
        if not is_instantiated(self.subestimator):
            return False
        return True

    def fit(self, X: np.array, y: np.array) -> None:
        if self.score_method not in self.available_metrics.keys():
            raise ValueError(f"Scoring must be one of: {self.available_metrics.keys()}")
        check_X_y(X=X, y=y)
        self.X_, self.y_ = X, y
        self.grid_ = BayesSearchCV(
            self.subestimator(), 
            self.param_grid, 
            n_jobs=-1, 
            scoring=self.score_method,
            n_iter=self.n_iter, 
            cv=3, 
            verbose=True)
        self.grid_.fit(X, y)
        self.n_targets_ = len(np.unique(y))
        self.subestimator = self.grid_.best_estimator_

    def predict(self, X: np.array) -> None:
        check_array(X)
        return self.grid_.predict(X)

    def predict_proba(self, X: np.array):
        self.check_is_fitted()
        check_array(X)
        return self.subestimator.predict_proba(X)

    def decision_function(self, X: np.array):
        self.check_is_fitted()
        check_array(X)
        return self.subestimator.decision_function(X)

    def compute_metric(self, metric: Callable, X: np.array, y: np.array):
        return metric(y, self.predict(X))

    def score(self, X: np.array, y: np.array):
        return self.compute_metric(self.available_metrics[self.score_method], X, y)


class BestDummyClassifier(BaseClassifier):
    sklearn_estimator = DummyClassifier
    param_grid = {
        "strategy": ["most_frequent", "uniform"],
    }
    n_iter = 8

class BestKNeighborsClassifier(BaseClassifier):
    sklearn_estimator = KNeighborsClassifier
    param_grid = {
        'n_neighbors': (1, 20),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
    }
    n_iter = 25


class BestSVC(BaseClassifier):
    sklearn_estimator = SVC
    param_grid = {
        'C': (1e-4, 1e+4, 'log-uniform'),
        'gamma': (1e-5, 1e+1, 'log-uniform'),
        'degree': (1, 3),
        'kernel': ['linear', 'poly', 'rbf'],
        'class_weight': ['balanced', None],
    }
    n_iter = 50


class BestRidgeClassifier(BaseClassifier):
    sklearn_estimator = RidgeClassifier
    param_grid = {
        'alpha': (1e-4, 1e+0, 'log-uniform'),
        'class_weight': ['balanced', None],
    }
    n_iter = 15


class BestRandomForestClassifier(BaseClassifier):
    sklearn_estimator = RandomForestClassifier
    param_grid = {
        'n_estimators': (50, 500),
        'max_depth': (5, 50),
        'max_features':  (1e-3, 1e+0, 'log-uniform'),
        'min_samples_split': (2, 5),
        'min_samples_leaf': (1, 5),
        'class_weight': ['balanced', None],
    }
    n_iter = 30