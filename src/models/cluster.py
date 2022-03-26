import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.utils.validation import check_array

from .base import BaseModel
from ..utils import get_array_counts


class BestKMeans(BaseModel):

    sklearn_estimator = KMeans
    available_metrics = {
        "calinski_harabasz": calinski_harabasz_score,
        "silhouette": silhouette_score
    }

    def __init__(self, k_min: int = 2, k_max: int = 6):
        super().__init__()
        self.k_min = k_min
        self.k_max = k_max
        self.scores = None
        self.models = None

    def fit(self, X: np.array, y: None = None) -> None:
        super().fit(X, y)
        check_array(X)
        self.models = {}
        self.scores = {}
        for k in range(self.k_min, self.k_max):
            model, model_name = self.estimator(n_clusters=k, random_state=0), k
            y_pred = model.fit_predict(X)
            self.models[model_name] = model
            self.scores[model_name] = {
                name: metric(X, y_pred) for name, metric in self.available_metrics.items()
            }

    def is_fitted(self) -> bool:
        if self.models is None or self.scores is None:
            return False
        return True

    def predict(self, X: np.array, k: int, return_counts: bool = False):
        super().predict(X)
        y_pred = self.models[k].predict(X)
        if return_counts:
            return get_array_counts(y_pred)
        return y_pred
