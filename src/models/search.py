import numpy as np
import pandas as pd

from sklearn.utils.validation import check_X_y, NotFittedError

from .base import BaseModel
from .classify import BaseClassifier, BestKNeighborsClassifier, BestSVC, BestRandomForestClassifier, BestRidgeClassifier


class AlreadyFittedError(Exception):
    pass


class ClassifierSearch(BaseModel):

    available_metrics = BaseClassifier.available_metrics
    sklearn_estimator = None

    def __init__(self, score_method: str, *args) -> None:

        self.score_method = score_method

        self._fitted = False
        self._results = None

        self.classifiers = [
            BestKNeighborsClassifier(score_method=self.score_method),
            BestSVC(score_method=self.score_method),
            BestRandomForestClassifier(score_method=self.score_method),
            BestRidgeClassifier(score_method=self.score_method)
        ]
        for clf in args:
            if not isinstance(clf, BaseClassifier):
                raise TypeError(f"Models must inherit from {BaseClassifier}")
            self.classifiers.append(clf)

    def is_fitted(self):
        return self._fitted

    def fit(self, X: np.array, y: np.array) -> None:
        check_X_y(X, y)
        if self.is_fitted():
            raise AlreadyFittedError("Fit method has already been called for ClassifierSearch")
        if self.score_method not in self.available_metrics.keys():
            raise ValueError(f"Scoring must be one of: {self.available_metrics.keys()}")

        n_fit, n_remain = 0, len(self.classifiers)
        for clf in self.classifiers:
            clf.fit(X, y)
            n_fit += 1
            n_remain -= 1
            print(f"Finished fitting model: {str(clf)}")
            print(f"Models Fit: {n_fit}\nModels Remaining: {n_remain}\n")
        self._fitted = True

    @property
    def results_(self):
        return self._results

    @results_.setter
    def results_(self, df: pd.DataFrame):
        self._results = df

    @property
    def subestimator_(self):
        if self.results_ is None:
            return None
        return self.classifiers[self.results_.index.get_loc(self.results_[self.score_method].idxmax())]

    def eval(self, X: np.array, y: np.array):
        check_X_y(X, y)
        if not self.is_fitted():
            raise NotFittedError
        results = {}
        for clf in self.classifiers:
            results[str(clf)] = {n: clf.compute_metric(m, X, y) for n, m in self.available_metrics.items()}
        self.results_ = pd.DataFrame.from_dict(results, orient='index')

    def score(self, X: np.array, y: np.array):
        check_X_y(X, y)
        if self.subestimator_ is None:
            self.eval(X, y)
        return self.subestimator_.score(X, y)

    def predict(self, X: np.array) -> None:
        pass

    def get_params(self, deep=True):
        return {"score_method": self.score_method}

