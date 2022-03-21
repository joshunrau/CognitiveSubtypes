import math

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

from .data import Dataset
from .utils import flatten_list

def plot_kmeans_elbow(data: Dataset, metric: str = 'distortion') -> None:
    
    available_metrics = ['distortion', 'calinski_harabasz', 'silhouette']
    if metric not in available_metrics:
        raise ValueError
    
    model = KMeans()
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