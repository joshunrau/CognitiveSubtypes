import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .base import Model
from ..features.cognitive import CognitiveFeatures

class ClusterModel(Model):
    
    features = CognitiveFeatures()
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("cluster", DBSCAN(eps=2, min_samples=15))
    ])
    
    def __init__(self):
        self.classes = self.pipeline.fit_predict(self.features)
    
    def __str__(self):
        msg = []
        for cls, cnt in np.array(np.unique(self.classes, return_counts=True)).T:
            msg.append(f"Class {cls}: {cnt}")
        return "\n".join(msg)

    