from abc import ABC
from collections import OrderedDict

import numpy as np
import pandas as pd

from .data import Dataset
from .utils import apply_dict_keys, camel_case_split, is_number


class BaseTable(ABC):
    pass


class SummaryTable:
    
    def __init__(self, data: Dataset, categorical_vars: list, continuous_vars: list):
        
        stats = OrderedDict()
        
        for var in categorical_vars:
            stats[var] = dict.fromkeys(["N", "Mean/Percent", "SD"], "")
            values = data.df[var].to_numpy()
            levels = np.array(np.unique(values, return_counts=True)).T
            assert levels.ndim == 2 and levels.shape[1] == 2
            for i in range(levels.shape[0]):
                level, count = levels[i]
                stats[level] = {
                    "N": count,
                    "Mean/Percent": count/len(values) * 100,
                    "SD": ""
                }
                
        for var in continuous_vars:
            values = data.df[var].to_numpy()
            stats[var] = {
                "N": sum(np.isnan(values) == False),
                "Mean/Percent": np.mean(values),
                "SD": np.std(values)
            }
        
        self.contents = pd.DataFrame.from_dict(apply_dict_keys(stats, camel_case_split), orient='index')
        self.contents = self.contents.applymap(lambda x: round(x, 2) if is_number(x) else x, na_action="ignore")