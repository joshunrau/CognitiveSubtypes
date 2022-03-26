from abc import ABC, abstractmethod
from typing import Type, Union, TypeVar

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, NotFittedError

Estimator = TypeVar("Estimator", bound=BaseEstimator)


class BaseModel(ABC):

    def __init__(self) -> None:
        self.X_ = None
        self.y_ = None
        self._subestimator = self.sklearn_estimator

    def __str__(self) -> str:
        return str(self.subestimator)

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
    def subestimator(self) -> Estimator | Type[Estimator]:
        return self._subestimator

    @abstractmethod
    def fit(self, X: np.array, y: Union[None, np.array]) -> None:
        self.X_ = X
        self.y_ = y

    @abstractmethod
    def predict(self, X: np.array) -> None:
        self.check_is_fitted()
        check_array(X)

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

