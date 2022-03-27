import os
from abc import ABC, abstractmethod
from copy import deepcopy

import pandas as pd

from ..data.dataset import Dataset
from ..filepaths import PATH_RESULTS_DIR
from ..utils import camel_case_split


class Table(ABC):

    def __init__(self, data: Dataset) -> None:
        self.data = data
    
    def __str__(self) -> str:
        return self.__class__.__name__
    
    @property
    @abstractmethod
    def contents(self) -> pd.DataFrame:
        pass

    @property
    def path(self):
        return os.path.join(PATH_RESULTS_DIR, "tables", str(self) + '.csv')

    def save(self):
        self.contents.to_csv(self.path)


class GroupedSummaryTable(Table):
    
    def __init__(self, data: Dataset, group_var: str) -> None:
        
        super().__init__(data)
    
        categorical_vars = ["sex"]
        continuous_vars = ["age"] + data.cognitive_feature_names
        
        df = deepcopy(data.df)
        df = df.merge(pd.get_dummies(df[categorical_vars], prefix="", prefix_sep=""), left_index=True, right_index=True)
        
        dummy_vars = []
        for variable in categorical_vars:
            if len(df[variable].unique()) != 2:
                raise ValueError
            dummy_vars.append(df[variable].value_counts().idxmax())

        levels = list(df[group_var].value_counts().index)
        stats = ["Count", "Mean/Percent", "SD"]

        self._contents = []
        for value in levels:
        
            subset_df = df[df[group_var] == value]

            sum_level = subset_df[[group_var] + dummy_vars].groupby(group_var).sum()
            total_var = subset_df[[group_var] + dummy_vars].groupby(group_var).count()
            percent_level = sum_level / total_var * 100
            categorical_stats = pd.concat([total_var, percent_level]).set_axis(stats[:2]).T
            categorical_stats = round(categorical_stats, 3)
            categorical_stats[stats[2]] = ""

            total_var = subset_df[[group_var] + continuous_vars].groupby(group_var).count()
            mean_var = subset_df[[group_var] + continuous_vars].groupby(group_var).mean()
            std_var = subset_df[[group_var] + continuous_vars].groupby(group_var).std()
            continuous_stats = pd.concat([total_var, mean_var, std_var]).set_axis(stats).T
            continuous_stats = round(continuous_stats, 3)

            self._contents.append(pd.concat([categorical_stats, continuous_stats]))
        
        header = pd.MultiIndex.from_product([levels, stats])
        self._contents = pd.concat(self._contents, axis=1)
        self._contents = self._contents.convert_dtypes()
        self._contents.columns = header
        self._contents.index = self._contents.index.map(camel_case_split)
    
    @property
    def contents(self):
        return self._contents


class PatientsVsControlsTable(GroupedSummaryTable):
    def __init__(self, data: Dataset) -> None:
        super().__init__(data, group_var = "subjectType")