import math

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

from ..data.dataset import Dataset
from ..models.cluster import BestKMeans
from ..utils import flatten_list


def plot_kmeans_scores(model: BestKMeans):
    
    k_values = list(model.scores.keys())
    calinski_harabasz_values = [x['calinski_harabasz'] for x in model.scores.values()]
    silhouette_values = [x['silhouette'] for x in model.scores.values()]
    assert len(k_values) == len(calinski_harabasz_values) == len(silhouette_values)

    fig, ax1 = plt.subplots(dpi=100)

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


def plot_kmeans_elbow(data: Dataset, metric: str = 'distortion') -> None:
    
    available_metrics = ['distortion', 'calinski_harabasz', 'silhouette']
    if metric not in available_metrics:
        raise ValueError
    
    model = KMeans(random_state=0)
    visualizer = KElbowVisualizer(model, k=(2, 6), timings=False, metric=metric)
    visualizer.fit(data.cognitive)

def plot_distributions(data: Dataset, variables: list) -> None:
    
    n_rows = math.ceil(len(variables) / 2)

    fig, axes = plt.subplots(nrows=n_rows, ncols=2, dpi=100)

    for i, ax in enumerate(axes.flat):
        try:
            sns.histplot(data=data.train.df, x=variables[i], ax=ax)
        except IndexError:
            break
        ax.set_title(variables[i])
    
    fig.set_figwidth(16)
    fig.set_figheight(8 * n_rows)
    fig.tight_layout()


def plot_transforms():
    
    data = Dataset()
    transformed_data = Dataset()
    transformed_data.apply_transforms()
    
    variables = flatten_list([x for x in flatten_list(transformed_data.transforms) if isinstance(x, list)])
    
    n_rows = len(variables)
    fig, axes = plt.subplots(nrows=n_rows, ncols=2, dpi=100)
    
    for i in range(n_rows):
        sns.histplot(data=data.train.df, x=variables[i], ax=axes[i][0])
        sns.histplot(data=transformed_data.train.df, x=variables[i], ax=axes[i][1])
    
    fig.set_figwidth(16)
    fig.set_figheight(8 * n_rows)
    fig.tight_layout()