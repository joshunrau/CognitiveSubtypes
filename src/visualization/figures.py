import os
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import seaborn as sns

from ..data.dataset import Dataset
from ..filepaths import PATH_RESULTS_DIR
from ..utils import camel_case_split

class Figure(ABC):

    def __str__(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def plot(self):
        pass

    @property
    def path(self):
        return os.path.join(PATH_RESULTS_DIR, "figures", str(self) + '.jpg')

    def save(self):
        plt.savefig(self.path)


class DataTransformFigure(Figure):

    def __init__(self) -> None:
        self.raw_data = Dataset.load()
        self.transformed_data = Dataset.load()
        self.transformed_data.apply_transforms()
        self.transformed_data.apply_scaler()
    
    def plot(self, **kwargs):
        variables = self.raw_data.cognitive_feature_names
        n_rows = len(variables)
        fig, axes = plt.subplots(nrows=n_rows, ncols=2, **kwargs)
        for i in range(n_rows):
            sns.histplot(data=self.raw_data.df, x=variables[i], ax=axes[i][0])
            sns.histplot(data=self.transformed_data.df, x=variables[i], ax=axes[i][1])
            axes[i][0].set(xlabel=camel_case_split(variables[i]))
            axes[i][1].set(xlabel=camel_case_split(variables[i]))
        fig.set_figwidth(12)
        fig.set_figheight(2 * n_rows)
        fig.tight_layout()


class KMeansScoresFigure(Figure):

    def __init__(self, model) -> None:
        self.model = model

    def plot(self, **kwargs):
        k_values = list(self.model.scores.keys())
        silhouette_values = [x['silhouette'] for x in self.model.scores.values()]
        fig, ax = plt.subplots(**kwargs)
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel("Silhouette Coefficient")
        ax.plot(k_values, silhouette_values)
        ax.scatter(k_values, silhouette_values)
        ax.set(xticks=k_values)
        ax.grid()
        fig.tight_layout()
