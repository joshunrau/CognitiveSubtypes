from abc import ABC, abstractmethod

from ..data.dataset import Dataset
from ..visualization.plots import plot_distributions

class Features(ABC):
    
    def __init__(self):
        self.df = Dataset.load()
        
    def __get__(self, obj, objtype=None):
        return self.df[self.names].to_numpy()
    
    def __str__(self):
        return "\n".join([
            "Feature Names",
            "-" * max([len(s) for s in self.names]),
            "\n".join(self.names)
        ])
    
    @property
    @abstractmethod
    def names(self):
        pass
    
    def plot_distributions(self):
        plot_distributions(self.df, self.names)