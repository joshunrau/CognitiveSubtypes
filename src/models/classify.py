import warnings
from abc import abstractmethod

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_array, check_is_fitted, NotFittedError

from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real
from .base import BaseModel


class BaseClassifier(BaseModel):

    available_metrics = {
        'accuracy': accuracy_score,
        'balanced_accuracy': balanced_accuracy_score,
        'roc_auc': roc_auc_score
    }

    prob_metrics = ['roc_auc']

    fs_param_grid = {}

    def __init__(self, score_method: str = 'accuracy') -> None:
        super().__init__()
        self._pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ("clf", self.sklearn_estimator())
        ])
        self._score_method = score_method

    def __str__(self) -> str:
        return str(self.sklearn_estimator.__name__)
    
    @property
    @abstractmethod
    def clf_param_grid(self) -> dict:
        pass

    @property
    def param_grid(self) -> dict:
        return self.fs_param_grid | self.clf_param_grid
    
    @property
    @abstractmethod
    def n_iter(self) -> int:
        pass

    @property
    def pipeline(self) -> Pipeline:
        return self._pipeline

    @pipeline.setter
    def pipeline(self, pipeline: Pipeline) -> None:
        if not isinstance(pipeline, Pipeline):
            raise TypeError
        self._pipeline = pipeline

    def is_fitted(self) -> bool:
        try:
            check_is_fitted(self.grid_)
        except (AttributeError, NotFittedError):
            return False
        return True

    def fit(self, X: np.ndarray, y: np.ndarray, surpress_warnings = True, **kwargs) -> None:
        super().fit(X, y)
        if self._score_method not in self.available_metrics.keys():
            raise ValueError(f"Scoring must be one of: {self.available_metrics.keys()}")
        self.grid_ = BayesSearchCV(
            self.pipeline,
            self.param_grid,
            n_jobs=-1,
            scoring=self._score_method,
            n_iter=self.n_iter,
            cv=5,
            **kwargs)
        print("Begin fitting best classifier for model: " + str(self))
        if surpress_warnings:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.grid_.fit(X, y)
        else:
            self.grid_.fit(X, y)
        self.n_targets_ = len(np.unique(y))
        self.best_estimator_ = self.grid_.best_estimator_
        self.best_params_ = self.grid_.best_params_
        self.best_score_ = self.grid_.best_score_
        self.classes_ = self.grid_.classes_
        self.n_features_in_ = self.grid_.n_features_in_
        print("Done!")
        print(f"{self._score_method}: {self.best_score_}")

    def predict(self, X: np.ndarray) -> None:
        self.check_is_fitted()
        check_array(X)
        return self.grid_.predict(X)

    def get_y_score(self, X: np.ndarray):
        self.check_is_fitted()
        check_array(X)
        if len(self.classes_) != 2:
            raise ValueError("This method was created for binary classes")
        try:
            return self.grid_.predict_proba(X)[:, 1]
        except AttributeError:
            return self.grid_.decision_function(X)

    def score(self, X: np.ndarray, y: np.ndarray):
        return self.grid_.score(X, y)


class BestKNeighborsClassifier(BaseClassifier):
    sklearn_estimator = KNeighborsClassifier
    clf_param_grid = {
        'clf__n_neighbors': Integer(1, 20),
        'clf__weights': Categorical(['uniform', 'distance']),
        'clf__metric': Categorical(['euclidean', 'manhattan', 'minkowski']),
    }
    n_iter = 25


class BestSVC(BaseClassifier):
    sklearn_estimator = SVC
    clf_param_grid = {
        'clf__C': Real(1e-2, 1e+3, 'log-uniform'),
        'clf__gamma': Real(1e-4, 1e+1, 'log-uniform'),
        'clf__degree': Integer(1, 3),
        'clf__kernel': Categorical(['linear', 'poly', 'rbf']),
        'clf__probability': Categorical([True])
    }
    n_iter = 50


class BestRidgeClassifier(BaseClassifier):
    sklearn_estimator = RidgeClassifier
    clf_param_grid = {
        'clf__alpha': Real(1e-4, 1e+0, 'log-uniform'),
        'clf__class_weight': Categorical(['balanced'])
    }
    n_iter = 15


class BestRandomForestClassifier(BaseClassifier):
    sklearn_estimator = RandomForestClassifier
    clf_param_grid = {
        'clf__n_estimators': Integer(50, 500),
        'clf__max_depth': Integer(5, 50),
        'clf__max_features': Real(1e-2, 1e+0, 'log-uniform'),
        'clf__min_samples_split': Integer(2, 5),
        'clf__min_samples_leaf': Integer(1, 5),
        'clf__class_weight': Categorical(['balanced'])
    }
    n_iter = 40


class BestGradientBoostingClassifier(BaseClassifier):
    sklearn_estimator = GradientBoostingClassifier
    clf_param_grid = {
        "clf__loss": Categorical(["deviance"]),
        "clf__learning_rate": Real(1e-3, 5e-1, 'log-uniform'),
        "clf__min_samples_split": Real(0.1, 0.9, 'log-uniform'),
        "clf__min_samples_leaf": Real(0.1, 0.5, 'log-uniform'),
        "clf__max_depth": Integer(2, 10),
        "clf__max_features": Categorical(["log2","sqrt"]),
        "clf__criterion": Categorical(["friedman_mse",  "squared_error"]),
        "clf__subsample": Real(0.5, 1, 'log-uniform')
    }
    n_iter = 50


class ClassifierSearch:
    
    def __init__(self):
        self.classifiers = []
        for clf in [BestKNeighborsClassifier, BestRidgeClassifier, BestRandomForestClassifier]:
            self.classifiers.append(clf(score_method='roc_auc'))
        self.model_names = ['KNN', 'RDG', 'RFC']
        self.roc_auc_scores = None
        
    def fit(self, data):
        self.roc_auc_scores = {"Train": [], "Test": []}
        for clf in self.classifiers:
            x_train = data.train.df[data.imaging_feature_names].to_numpy()
            x_test = data.test.df[data.imaging_feature_names].to_numpy()
            clf.fit(x_train, data.train.target, surpress_warnings=True, verbose=False)
            self.roc_auc_scores["Train"].append(clf.best_score_)
            self.roc_auc_scores["Test"].append(clf.score(x_test, data.test.target))
        self.roc_auc_scores = pd.DataFrame(self.roc_auc_scores, index=self.model_names)
        self.best_classifier = self.classifiers[self.model_names.index(self.roc_auc_scores["Test"].idxmax())]

  
def get_feature_importances(best_random_forest, feature_names):
    
    forest = best_random_forest.best_estimator_.named_steps['clf']
    if len(feature_names) != len(forest.feature_importances_):
        raise ValueError
    forest_importances = {k:v for k, v in zip(feature_names, forest.feature_importances_)}
    
    regional_importances = {}
    for feature in forest_importances:
        for measure in ['area', 'thickness', 'volume']:
            if feature.startswith(measure):
                try:
                    regional_importances[feature.replace(measure, '')][measure] = forest_importances[feature]
                except KeyError:
                    regional_importances[feature.replace(measure, '')] = {}
                    regional_importances[feature.replace(measure, '')][measure] = forest_importances[feature]
    regional_importances = pd.DataFrame.from_dict(regional_importances).T
    regional_importances.index = regional_importances.index.map(lambda x: x.lower().replace("left", "lh_").replace("right", "rh_"))
    regional_importances = regional_importances.rename(lambda x: x.capitalize(), axis=1)
    return regional_importances