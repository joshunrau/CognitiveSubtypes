import os

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.legend import Legend
from yellowbrick.classifier import PrecisionRecallCurve

from ..data.dataset import Dataset
from ..filepaths import PATH_RESULTS_DIR
from ..utils import camel_case_split

class Figure(ABC):

    fontsize=12

    def __str__(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def plot(self):
        pass

    @property
    def path(self):
        return os.path.join(PATH_RESULTS_DIR, "figures", str(self) + '.jpg')

    def save(self):
        plt.savefig(self.path, dpi=300, bbox_inches='tight')


class DataTransformFigure(Figure):

    def __init__(self) -> None:
        self.raw_data = Dataset.load()
        self.transformed_data = Dataset.load()
        self.transformed_data.apply_transforms()
        self.transformed_data.apply_scaler()
    
    def plot(self):
        variables = self.raw_data.cognitive_feature_names
        n_rows = len(variables)
        fig, axes = plt.subplots(nrows=n_rows, ncols=2)
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

    def plot(self):
        k_values = list(self.model.scores.keys())
        silhouette_values = [x['silhouette'] for x in self.model.scores.values()]
        fig, ax = plt.subplots()
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel("Silhouette Coefficient")
        ax.plot(k_values, silhouette_values)
        ax.scatter(k_values, silhouette_values)
        ax.set(xticks=k_values)
        ax.grid()
        fig.tight_layout()


class PRCurveFigure(Figure):

    def __init__(self, clf, data) -> None:
        self.clf = clf
        self.data = data

    def plot(self, title = " "):

        try:
            estimator = self.clf.subestimator
        except AttributeError:
            estimator = self.clf
        
        viz = PrecisionRecallCurve(
            estimator,
            classes=self.clf.classes_,
            colors=["purple", "cyan", "blue"],
            per_class=True,
            micro=False,
            title=title
        )
        viz.fit(self.data.train.imaging, self.data.train.target)
        viz.score(self.data.test.imaging, self.data.test.target)
        viz.finalize()


def violin_plot(data):
    
    x = "class"
    hue = "subjectType"
    fontsize = 12

    x_labels = {
        "0": "HSD",
        "1": "SPM",
        "2": "GS"
    }

    hue_labels = {
        "patient": "Patients",
        "control": "Controls"
    }

    variables = data.cognitive_feature_names

    fig, axes = plt.subplots(ncols=4, nrows=2, sharey=True, figsize=(12, 8))
    axes = axes.flatten()
    for i in range(len(variables)):
        sns.violinplot(data=data.df, x=x, y=variables[i], hue=hue, split=True,
            palette=sns.color_palette('colorblind'), ax=axes[i], showfliers = False, dodge=False)
        axes[i].set_title(camel_case_split(variables[i]), fontsize=fontsize)
        axes[i].set_xlabel(None)
        axes[i].set_ylabel(None)
        xticklabels = [txt.get_text() for txt in axes[i].get_xticklabels()]
        axes[i].set_xticklabels([x_labels[lab] for lab in xticklabels])
        axes[i].legend().set_visible(False)

    axes[0].set_ylabel("Z Scores")
    axes[4].set_ylabel("Z Scores")
    axes[7].grid(None)
    axes[7].set_xticklabels([])

    lines, hueticklabels = axes[0].get_legend_handles_labels()
    hueticklabels = [hue_labels[lab] for lab in hueticklabels]
    axes[7].add_artist(Legend(axes[7], lines,  hueticklabels, title="Group", 
        fontsize=fontsize, ncol=1, loc='center', frameon=False))
    for dim in ['top', 'right', 'bottom', 'left']:
        axes[7].spines[dim].set_visible(False)
    fig.subplots_adjust(wspace=0)
    fig.tight_layout()
    plt.savefig(os.path.join(PATH_RESULTS_DIR, "figures", 'violin.jpg'), dpi=300, bbox_inches='tight')