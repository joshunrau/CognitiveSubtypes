import pandas as pd

from src.models.base import BaseClassifier, BaseModel
from src.models.classify import BestKNeighborsClassifier, BestSVC, BestRandomForestClassifier, BestRidgeClassifier


class ClassifierSearch(BaseModel):

    available_score_methods = BaseClassifier.available_score_methods

    def __init__(self, score_method: str, *args) -> None:

        self.score_method = score_method

        self.classifiers = [
            BestKNeighborsClassifier(score_method=self.score_method),
            BestSVC(score_method=self.score_method),
            BestRandomForestClassifier(score_method=self.score_method),
            BestRidgeClassifier(score_method=self.score_method)
        ]

        for model in args:
            if not isinstance(model, BaseClassifier):
                raise TypeError(f"Models must inherit from {BaseClassifier}")
            self.classifiers.append(model)

    def __str__(self) -> str:
        return str(self.estimator)
    
    @property
    @classmethod
    def estimator(self):
        pass

    def fit(self, data) -> None:

        super().fit(data)
        
        if self.score_method not in self.available_score_methods.keys():
            raise ValueError(f"Scoring must be one of: {self.available_score_methods.keys()}")

        self.results = {}
        n_fit, n_remain = 0, len(self.classifiers)
        for clf in self.classifiers:
            clf.fit(data)
            scores = {n: clf.compute_metric(m, data) for n, m in self.available_score_methods.items()}
            self.results[str(clf)] = scores
            n_fit += 1
            n_remain -= 1
            print("Finished fitting model: " + str(clf))
            for metric, score in scores.items():
                print(metric.replace("_", " ") + ": " + str(round(score, 3)))
            print(f"Models Fit: {n_fit}\nModels Remaining: {n_remain}\n")

        self.results = pd.DataFrame.from_dict(self.results, orient='index')
    
    def predict(self, data) -> None:
        pass