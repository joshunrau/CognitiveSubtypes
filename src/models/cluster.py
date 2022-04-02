import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from .base import BaseModel
from ..utils import get_array_counts


class BestKMeans(BaseModel):

    sklearn_estimator = KMeans

    available_metrics = {
        "calinski_harabasz": calinski_harabasz_score,
        "davies_bouldin": davies_bouldin_score,
        "silhouette": silhouette_score
    }

    def __init__(self, k_min: int = 2, k_max: int = 6):
        super().__init__()
        self._estimator = self.sklearn_estimator
        self._k_min = k_min
        self._k_max = k_max

    def fit(self, X: np.ndarray, y: None = None) -> None:
        super().fit(X, y)
        self.models_ = {}
        self.scores_ = {}
        for k in range(self._k_min, self._k_max + 1):
            model, model_name = self._estimator(n_clusters=k), k
            y_pred = model.fit_predict(X)
            self.models_[model_name] = model
            self.scores_[model_name] = {
                name: metric(X, y_pred) for name, metric in self.available_metrics.items()
            }

    def is_fitted(self) -> bool:
        if self.models_ is None or self.scores_ is None:
            return False
        return True

    def predict(self, X: np.array, k: int, return_counts: bool = False):
        super().predict(X)
        y_pred = self.models_[k].predict(X)
        if return_counts:
            return get_array_counts(y_pred)
        return y_pred
