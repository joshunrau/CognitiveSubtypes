from abc import ABC, abstractmethod
from typing import Type

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_X_y, NotFittedError


class BaseModel(ABC):

    @property
    @abstractmethod
    def sklearn_estimator(self) -> Type[BaseEstimator]:
        pass

    def check_is_fitted(self) -> None:
        if not self.is_fitted():
            raise NotFittedError("Object must be fitted before method call")

    @abstractmethod
    def is_fitted(self) -> bool:
        pass

    @property
    @abstractmethod
    def available_metrics(self) -> dict:
        pass

    @abstractmethod
    def fit(self, X: np.ndarray, y: None | np.ndarray = None) -> None:
        if y is None:
            check_array(X)
        else:
            check_X_y(X, y)
        self.X_ = X
        self.y_ = y

    @abstractmethod
    def predict(self, X: np.ndarray) -> None:
        self.check_is_fitted()
        check_array(X)

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
