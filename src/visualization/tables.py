import numpy as np
import pandas as pd

from collections import OrderedDict

from ..data.dataset import Dataset

class Variable:
    
    def __init__(self, name, table_name, vtype="continuous"):
        
        self.name = name
        self.table_name = table_name
        
        if vtype not in ["continuous", "categorical"]:
            raise TypeError
        
        self.vtype = vtype
        
    def is_categorical(self):
        return self.vtype == "categorical"


def get_summary_statistics():
    
    df = Dataset.load()
    
    variables = [
        Variable(name="age", table_name="Age"),
        Variable(name="sex", table_name="Sex", vtype="categorical"),
        Variable(name="meanReactionTimeTest", table_name="Mean Score Reaction Time Test"),
        Variable(name="timeTrailMakingTestA", table_name="Time Trail Making Test Part A"),
        Variable(name="timeTrailMakingTestB", table_name="Time Trail Making Test Part B"),
        Variable(name="accuracyTowerTest", table_name="Accuracy Tower Test"),
        Variable(name="accuracySymbolDigitTest", table_name="Accuracy Symbol Digit Test"),
        Variable(name="incorrectPairsMatchingTask", table_name="Incorrect Pairs Matching Task"),
        Variable(name="prospectiveMemoryTask", table_name="Result Prospective Memory Task")
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
    
    return pd.DataFrame.from_dict(stats, orient='index', columns=["N", "Mean/Percent", "SD"])