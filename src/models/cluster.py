from typing import Callable

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .base import BaseModel


class BestKMeans(BaseModel):
    """ methods accept object of a Dataset class """
    available_score_funcs = [silhouette_score]
    estimator = KMeans

    def __init__(self, score_func: Callable = silhouette_score) -> None:
        super().__init__(score_func)
    
    def fit(self, data, k_min: int = 2, k_max: int = 6) -> None:
        
        super().fit(data)
        
        self.scores = {}

        best_score = -1
        best_model = None

        for k in range(k_min, k_max):
            model = self.estimator(n_clusters=k)
            y_pred = model.fit_predict(self._data.cognitive)
            score = self.score_func(self._data.cognitive, y_pred)
            if score > best_score:
                best_model = model
                best_score = score
            self.scores[str(model)] = score
        self.estimator = best_model

    def predict(self, data) -> tuple:
        super().predict(data)
        y_train = self.estimator.predict(data.train.cognitive)
        y_test = self.estimator.predict(data.test.cognitive)
        return y_train, y_test

