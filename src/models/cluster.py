import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.utils.validation import NotFittedError

from .base import BaseModel
from ..utils import get_array_counts


class BestKMeans(BaseModel):
    
    estimator = KMeans
    available_metrics = {
        "calinski_harabasz": calinski_harabasz_score, 
        "silhouette": silhouette_score
    }

    def fit(self, data, k_min: int = 2, k_max: int = 6) -> None:

        super().fit(data)
        
        self.models = {}
        self.scores = {}

        for k in range(k_min, k_max):
            model, model_name = self.estimator(n_clusters=k, random_state=0), k
            y_pred = model.fit_predict(self._data.cognitive)
            self.models[model_name] = model
            self.scores[model_name] = {
                name: metric(self._data.cognitive, y_pred) for name, metric in self.available_metrics.items()
            }

    def predict(self, data, k: int) -> tuple:
        
        try:
            model = self.models[k]
        except AttributeError as err:
            raise NotFittedError from err

        y_train = model.predict(data.train.cognitive)
        y_test = model.predict(data.test.cognitive)
        return y_train, y_test
    
    def get_class_counts(self, data, k: int) -> pd.DataFrame:
        return get_array_counts(self.predict(data, k)[0])