from abc import ABC, abstractmethod
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from yellowbrick.classifier import ROCAUC
from yellowbrick.model_selection import FeatureImportances

from ..data.dataset import Dataset
from ..filepaths import PATH_RESULTS_DIR
from ..utils import camel_case_split


class Figure(ABC):

    @abstractmethod
    def plot(self):
        pass

    def save(self, filename):
        plt.savefig(os.path.join(PATH_RESULTS_DIR, "figures", filename), dpi=300, bbox_inches='tight')


class KMeansScores(Figure):

    def __init__(self, model) -> None:
        self.model = model

    def plot(self):

        k_values = list(self.model.scores_.keys())
        calinski_harabasz_values = [x['calinski_harabasz'] for x in self.model.scores_.values()]
        silhouette_values = [x['silhouette'] for x in self.model.scores_.values()]
        assert len(k_values) == len(calinski_harabasz_values) == len(silhouette_values)

        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel("Calinski-Harabasz Index", color=color)
        ax1.plot(k_values, calinski_harabasz_values, color=color)
        ax1.scatter(k_values, calinski_harabasz_values, color=color)
        ax1.set(xticks=k_values)
        ax1.tick_params(axis='y', labelcolor=color)

        color = 'tab:blue'
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel("Silhouette Coefficient", color=color)
        ax2.plot(k_values, silhouette_values, color=color)
        ax2.scatter(k_values, silhouette_values, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.grid(None)

        fig.tight_layout()


class AUCScores(Figure):

    def __init__(self, cs) -> None:
        self.cs = cs

    def plot(self):
        fig, axes = plt.subplots(ncols=2, sharey=True, figsize=(10, 4))
        p1 = axes[0].bar(self.cs.model_names, self.cs.roc_auc_scores["Train"], align='center', alpha=1, color="#408ec6")
        axes[0].bar_label(p1, label_type='edge', fmt='%.2f')
        axes[0].set_title("Training")
        axes[0].set_ylabel('AUC')
        axes[0].set_xticks((0, 1, 2), self.cs.model_names)
        p2 = axes[1].bar(self.cs.model_names, self.cs.roc_auc_scores["Test"], align='center', alpha=1, color="#1e2761")
        axes[1].bar_label(p2, label_type='edge', fmt='%.2f')
        axes[1].set_title("Validation")
        axes[1].set_xticks((0, 1, 2), self.cs.model_names)
        fig.subplots_adjust(top=1, wspace=.05)


class ROCCurve(Figure):

    def __init__(self, cs, data) -> None:
        self.cs = cs
        self.data = data
    
    def plot(self):
        viz = ROCAUC(self.cs.best_classifier.best_estimator_, binary=True, title=" ")
        viz.fit(self.data.train.imaging, self.data.train.target)
        viz.score(self.data.test.imaging, self.data.test.target)
        viz.finalize()


class Transforms(Figure):

    def plot(self):

        raw_data = Dataset.load()
        transformed_data = Dataset.load_preprocess()

        variables = raw_data.cognitive_feature_names
        nrows = len(variables)
        fig, axes = plt.subplots(nrows=nrows, ncols=2)
        
        for i in range(nrows):
            sns.histplot(data=raw_data.df, x=variables[i], ax=axes[i][0])
            axes[i][0].set(xlabel=camel_case_split(variables[i]))
            axes[i][0].set_ylabel(None)
        
        for i in range(nrows):
            sns.histplot(data=transformed_data.df, x=variables[i], ax=axes[i][1])
            axes[i][1].set(xlabel=camel_case_split(variables[i]))
            axes[i][1].set_ylabel(None)
        
        fig.subplots_adjust(wspace=0)
        fig.set_figwidth(8)
        fig.set_figheight(1.2 * len(variables))
        fig.tight_layout()


class ViolinPlot(Figure):

    def __init__(self, data) -> None:
        self.data = data
    
    def plot(self):

        x = "class"
        hue = "subjectType"
        fontsize = 12

        x_labels = {
            "1": "Low",
            "0": "High",
        }

        hue_labels = {
            "patient": "Patients",
            "control": "Controls"
        }

        variables = self.data.cognitive_feature_names

        fig, axes = plt.subplots(ncols=4, nrows=2, sharey=True, figsize=(12, 8))
        axes = axes.flatten()
        for i in range(len(variables)):
            sns.violinplot(data=self.data.df, x=x, y=variables[i], hue=hue, split=True,
                palette=sns.color_palette('colorblind'), ax=axes[i], showfliers = False, dodge=False)
            axes[i].set_title(camel_case_split(variables[i]), fontsize=fontsize)
            axes[i].set_xlabel(None)
            axes[i].set_ylabel(None)
            xticklabels = [txt.get_text() for txt in axes[i].get_xticklabels()]
            axes[i].set_xticklabels([x_labels[lab] for lab in xticklabels])
            axes[i].legend().set_visible(False)

        axes[0].set_ylabel("Z Scores")
        axes[4].set_ylabel("Z Scores")

        lines, labels = axes[0].get_legend_handles_labels()
        labels = [hue_labels[lab] for lab in labels]
        
        fig.legend(lines, labels, ncol=len(labels), loc='lower center', 
                fontsize=fontsize, borderaxespad=1)
        
        fig.subplots_adjust(wspace=0)


class TopFeatures(Figure):

    def __init__(self, model, data) -> None:
        self.model = model
        self.data = data
    
    def get(self):
        results = {}
        rf = self.model.best_estimator_.named_steps.clf
        for key, value in zip(self.data.imaging_feature_names, rf.feature_importances_):
            results[self.format_label(key)] = value
        return pd.DataFrame.from_dict(results, orient='index').sort_values(by=0, ascending=False)

    def plot(self):
        labels = [self.format_label(x) for x in self.data.imaging_feature_names]
        viz = FeatureImportances(self.model.best_estimator_.named_steps.clf, topn=20, labels=labels)
        viz.fit(self.data.train.imaging, self.data.train.target)
        viz.finalize()
    
    @classmethod
    def format_label(cls, label):
        label = camel_case_split(label)
        for s in ['inferior', 'middle', 'superior', 'anterior', 'posterior', 'rostral', 'caudal', 'lateral', 'medial']:
            label = cls.separate(label, s)
        return label

    @staticmethod
    def separate(s, substring):
        splt = s.lower().split(substring.lower())
        if len(splt) == 1:
            return s
        elif len(splt) != 2:
            raise ValueError(f"List contains more than two strings: {splt}")
        splt.insert(1, substring)
        return " ".join([x.strip() for x in splt]).title()