from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Callable

import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import balanced_accuracy_score, classification_report, make_scorer, roc_auc_score, silhouette_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from .data import Dataset


class ModelNotFittedError(Exception):
    pass


class BaseModel(ABC):

    def __init__(self, score_func: Callable) -> None:
        self._fitted = False
        self.score_func = score_func
    
    def check_fitted(self):
        if not self._fitted:
            raise ModelNotFittedError
    
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
        self._fitted = True
        if self.score_func not in self.available_score_funcs:
            raise ValueError(f"Scorer must be one of: {self.available_score_funcs}")
    
    @abstractmethod
    def predict(self, data: Dataset) -> None:
        self.check_fitted()


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

    def predict(self, data: Dataset) -> None:
        super().predict(data)
        return self.grid.predict(data.test.imaging)
    
    def classification_report(self, data: Dataset) -> None:
        print(classification_report(data.test.target, self.predict(data)))
    
    def compute_metric(self, metric: Callable, data: Dataset):
        return metric(data.test.target, self.predict(data))


class BestSVC(BaseClassifier):
    
    estimator = SVC
    param_grid = {
        'class_weight': ['balanced'],
        'C': [0.1, 1, 10, 100], 
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }

    def fit(self, data: Dataset) -> None:
        super().fit(data)


class BestKMeans(BaseModel):

    available_score_funcs = [silhouette_score]
    estimator = KMeans

    def __init__(self, score_func: Callable = silhouette_score) -> None:
        super().__init__(score_func)
    
    def fit(self, data: Dataset, k_min: int = 2, k_max: int = 6) -> None:
        
        super().fit(data)
        
        self.best_score = -1
        self.best_model = None
        self.scores = {}

        for k in range(k_min, k_max):
            model = self.estimator(n_clusters=k)
            y_pred = model.fit_predict(self._data.cognitive)
            score = self.score_func(self._data.cognitive, y_pred)
            if score > self.best_score:
                self.best_model = model
                self.best_score = score
            self.scores[str(model)] = score
    
    def predict(self, data: Dataset) -> tuple:
        super().predict(data)
        y_train = self.best_model.predict(data.train.cognitive)
        y_test = self.best_model.predict(data.test.cognitive)
        return y_train, y_test


