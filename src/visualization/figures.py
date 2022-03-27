import os
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt

from ..filepaths import PATH_RESULTS_DIR


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


class KMeansScores(Figure):

    def __init__(self, model) -> None:
        self.model = model

    def plot(self, **kwargs):
        k_values = list(self.model.scores.keys())
        calinski_harabasz_values = [x['calinski_harabasz'] for x in self.model.scores.values()]
        silhouette_values = [x['silhouette'] for x in self.model.scores.values()]
        assert len(k_values) == len(calinski_harabasz_values) == len(silhouette_values)

        fig, ax1 = plt.subplots(**kwargs)

        color = 'tab:red'
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel("Calinski-Harabasz Score", color=color)
        ax1.plot(k_values, calinski_harabasz_values, color=color)
        ax1.scatter(k_values, calinski_harabasz_values, color=color)
        ax1.set(xticks=k_values)
        ax1.tick_params(axis='y', labelcolor=color)

        color = 'tab:blue'
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel("Silhouette Score", color=color)
        ax2.plot(k_values, silhouette_values, color=color)
        ax2.scatter(k_values, silhouette_values, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.grid(None)

        fig.tight_layout()
