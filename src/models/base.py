from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np

from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import check_is_fitted, NotFittedError

from ..data.dataset import Dataset
from ..utils import is_instantiated

class BaseModel(ABC):
    
    def __init__(self) -> None:
        pass
        
    def __str__(self) -> str:
        return str(self.estimator)
    
    def check_is_fitted(self):
        if not self.is_fitted():
            raise NotFittedError
    
    def is_fitted(self):
        if not is_instantiated(self.estimator):
            return False
        return check_is_fitted
    
    @property
    @classmethod
    @abstractmethod
    def estimator(self):
        pass

    @abstractmethod
    def fit(self, data: Dataset) -> None:
        self._data = deepcopy(data)
    
    @abstractmethod
    def predict(self, data: Dataset) -> None:
        self.check_is_fitted()


class BaseClassifier(BaseModel):
    
    available_score_methods = {
        'accuracy': accuracy_score,
        'balanced_accuracy': balanced_accuracy_score
    }

    def __init__(self, score_method: str) -> None:
        self.score_method = score_method
    
    @property
    @classmethod
    @abstractmethod
    def param_grid(self):
        pass

    def fit(self, data) -> None:
        super().fit(data)
        if self.score_method not in self.available_score_methods.keys():
            raise ValueError(f"Scoring must be one of: {self.available_score_methods.keys()}")
        if self._data.target is None:
            raise ValueError
        self.grid = GridSearchCV(self.estimator(), self.param_grid, n_jobs=-1, scoring=self.score_method)
        self.grid.fit(self._data.train.imaging, self._data.train.target)
        self.n_targets = len(np.unique(self._data.train.target))
        self.estimator = self.grid.best_estimator_
    
    def predict(self, data) -> None:
        super().predict(data)
        return self.grid.predict(data.test.imaging)
    
    def predict_proba(self, data):
        self.check_is_fitted()
        if self.n_targets == 2:
            return self.estimator.predict_proba(data.test.imaging)[:, 1]
        return self.estimator.predict_proba(data.test.imaging)
    
    def decision_function(self, data):
        return self.estimator.decision_function(data.test.imaging)
    
    def compute_metric(self, func, data):
        return func(data.test.target, self.predict(data))
    
    def score(self, data):
        return self.compute_metric(self.available_score_methods[self.score_method], data)
        