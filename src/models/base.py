import math

from abc import ABC, abstractmethod
from ..data.dataset import Dataset

import matplotlib.pyplot as plt
import seaborn as sns

class BaseModel(ABC):

    def __init__(self) -> None:
        data = Dataset.load()
        self.df = data.df[[data.idvar] + self.feature_names]
        assert self.features.shape[1] == self.n_features
    
    @property
    @abstractmethod
    def pipeline(self):
        pass
    
    @property
    def features(self):
        return self.df[self.feature_names].to_numpy()
    
    @property
    @abstractmethod
    def feature_names(self):
        pass

    @property
    def n_features(self):
        return len(self.feature_names)
    
    def plot_feature_distributions(self, filepath=None):

        fig, axes = plt.subplots(nrows=math.ceil(self.n_features / 2), ncols=2, figsize=(16, 16), dpi=100, tight_layout=True)
        for i, ax in enumerate(axes.flat):
            try:
                feature = self.features[:, i]
            except IndexError:
                break
            sns.histplot(feature, ax=ax)
            ax.set_title(self.feature_names[i])
        fig.suptitle('Feature Distributions', fontsize=16)
        
        if filepath is not None:
            plt.savefig(filepath)
