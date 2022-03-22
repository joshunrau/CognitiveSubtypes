from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Callable

from sklearn.metrics import balanced_accuracy_score, classification_report, make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import check_is_fitted, NotFittedError

from ..data.dataset import Dataset
from ..utils import is_instantiated

class BaseModel(ABC):

    def __init__(self, score_func: Callable) -> None:
        self.score_func = score_func
    
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
    
    @property
    @classmethod
    @abstractmethod
    def available_score_funcs(self):
        pass

    @abstractmethod
    def fit(self, data: Dataset) -> None:
        self._data = deepcopy(data)
        if self.score_func not in self.available_score_funcs:
            raise ValueError(f"Scorer must be one of: {self.available_score_funcs}")
    
    @abstractmethod
    def predict(self, data: Dataset) -> None:
        self.check_is_fitted()


class BaseClassifier(BaseModel):
    
    available_score_funcs = [balanced_accuracy_score, roc_auc_score]
    def __init__(self, score_func: Callable = roc_auc_score) -> None:
        super().__init__(score_func)
    
    @property
    @classmethod
    @abstractmethod
    def param_grid(self):
        pass

    def fit(self, data: Dataset) -> None:
        super().fit(data)
        if self._data.target is None:
            raise ValueError
        self.grid = GridSearchCV(self.estimator(), self.param_grid, n_jobs=-1, scoring=make_scorer(self.score_func))
        self.grid.fit(self._data.train.imaging, self._data.train.target)
        self.estimator = self.grid.best_estimator_
    
    def predict(self, data: Dataset) -> None:
        super().predict(data)
        return self.grid.predict(data.test.imaging)
    
    def predict_proba(self, data: Dataset):
        self.check_is_fitted()
        try:
            return self.estimator.predict_proba(data.test.imaging)
        except AttributeError:
            return None
    
    def classification_report(self, data: Dataset) -> None:
        print(classification_report(data.test.target, self.predict(data)))
    
    def compute_metric(self, metric: Callable, data: Dataset):
        return metric(data.test.target, self.predict(data))
    
    def score(self, data: Dataset):
        self.check_is_fitted()
        return self.compute_metric(self.score_func, data)