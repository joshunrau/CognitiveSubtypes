from abc import ABC, abstractmethod
from typing import Type, Union, TypeVar

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, NotFittedError

Estimator = TypeVar("Estimator", bound=BaseEstimator)


class BaseModel(ABC):

    def __init__(self) -> None:
        self._X = None
        self._y = None
        self._estimator = self.sklearn_estimator

    def __str__(self) -> str:
        return str(self.estimator)

    def check_is_fitted(self) -> None:
        if not self.is_fitted():
            raise NotFittedError("Estimator must be fitted before method call")

    @abstractmethod
    def is_fitted(self) -> bool:
        pass

    @property
    @abstractmethod
    def available_metrics(self) -> dict:
        pass

    @property
    @abstractmethod
    def sklearn_estimator(self) -> Estimator | Type[Estimator]:
        pass

    @property
    def estimator(self) -> Estimator | Type[Estimator]:
        return self._estimator

    @abstractmethod
    def fit(self, X: np.array, y: Union[None, np.array]) -> None:
        self._X = X
        self._y = y

    @abstractmethod
    def predict(self, X: np.array) -> None:
        self.check_is_fitted()
        check_array(X)
