import os

import numpy as np
import pandas as pd

from collections import OrderedDict

from ..data.dataset import Dataset
from . import RESULTS_DIR
from .utils import camel_case_split, is_number

class Variable:
    
    def __init__(self, name, vtype="continuous"):
        
        self.name = name
        self.table_name = camel_case_split(name)
        
        if vtype not in ["continuous", "categorical"]:
            raise TypeError
        
        self.vtype = vtype
        
    def is_categorical(self):
        return self.vtype == "categorical"
    
def summary_statistics(df):
    
    variables = [
        Variable(name="age"),
        Variable(name="sex", vtype="categorical"),
        #Variable(name="meanReactionTimeTest"),
        #Variable(name="timeTrailMakingTestA"),
        #Variable(name="timeTrailMakingTestB"),
        #Variable(name="accuracyTowerTest"),
        #Variable(name="accuracySymbolDigitTest"),
        #Variable(name="incorrectPairsMatchingTask"),
        #Variable(name="prospectiveMemoryTask")
    ]
    
    stats = OrderedDict()
    
    for variable in variables:
        data = df[variable.name].to_numpy()
        if variable.is_categorical():
            stats[variable.table_name] = ["", "", ""]
            levels = np.array(np.unique(data, return_counts=True)).T
            assert levels.ndim == 2 and levels.shape[1] == 2
            for i in range(levels.shape[0]):
                level, count = levels[i]
                stats[level] = [count, count/len(data) * 100, ""]
        else:
            stats[variable.table_name] = [sum(np.isnan(data) == False), np.mean(data), np.std(data)]
    
    stats = pd.DataFrame.from_dict(stats, orient='index', columns=["N", "Mean/Percent", "SD"])
    stats = stats.applymap(lambda x: round(x, 2) if is_number(x) else x, na_action="ignore")
    stats.to_csv(os.path.join(RESULTS_DIR, "summary_statistics.csv"))