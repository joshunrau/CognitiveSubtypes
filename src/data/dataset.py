import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, StandardScaler

from .build import DataBuilder
from ..filepaths import PATH_DATA_DIR
from ..utils import camel_case_split


class Data:

    id_var = 'id'
    target_var = 'class'

    def __init__(self, df):
        self._df = df
        self._target = None
        self.df[self.target_var] = self._target

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, new_df):
        if not isinstance(new_df, pd.DataFrame):
            raise TypeError
        if not list(self.df.columns) == list(new_df.columns):
            raise ValueError
        self._df = new_df

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, value):
        if len(value) != len(self.df):
            raise ValueError
        self._target = value
        self.df[self.target_var] = value

    def get(self, names):
        return self.df[names].to_numpy()

    @property
    def cognitive_feature_names(self):
        return [
            "meanReactionTimeTest",
            "timeTrailMakingTestA",
            "timeTrailMakingTestB",
            "accuracyTowerTest",
            "accuracySymbolDigitTest",
            "incorrectPairsMatchingTask",
            "prospectiveMemoryTask"
        ]

    @property
    def imaging_feature_names(self):
        imaging_feature_names = []
        for col in self.df.columns:
            if col.startswith("area") or col.startswith("thickness") or col.startswith("volume"):
                imaging_feature_names.append(col)
        return imaging_feature_names

    @property
    def feature_names(self):
        return self.cognitive_feature_names + self.imaging_feature_names

    @property
    def cognitive(self):
        return self.get(self.cognitive_feature_names)

    @property
    def imaging(self):
        return self.get(self.imaging_feature_names)

    @property
    def features(self):
        return self.get(self.feature_names)


class Dataset(Data):

    def __init__(self, df):
        train_data, test_data = train_test_split(df, test_size=0.3, random_state=0)
        self.train, self.test = Data(train_data), Data(test_data)

    def apply_transforms(self):

        log_transformer = FunctionTransformer(np.log)
        power_transformer = PowerTransformer(standardize=False)

        self.transforms = [
            [log_transformer, ["meanReactionTimeTest", "timeTrailMakingTestA", "timeTrailMakingTestB"]],
            [power_transformer, ["accuracyTowerTest", "accuracySymbolDigitTest", "incorrectPairsMatchingTask"]]
        ]

        for transformer, variables in self.transforms:
            self.train.df[variables] = transformer.fit_transform(self.train.df[variables])
            self.test.df[variables] = transformer.transform(self.test.df[variables])

    def apply_scaler(self, scaler=StandardScaler()):
        self.train.df[self.train.feature_names] = scaler.fit_transform(self.train.features)
        self.test.df[self.test.feature_names] = scaler.transform(self.test.features)

    def summarize_by_class(self):
        if self.target is None:
            raise ValueError("Target for Dataset has not yet been set!")
        include = self.cognitive_feature_names + [self.target_var]
        df = self.df[include].groupby(self.target_var).mean().T
        df.index = df.index.map(camel_case_split)
        return df

    @property
    def df(self):
        if list(self.train.df.columns) != list(self.test.df.columns):
            raise AssertionError
        return pd.concat(objs=[self.train.df, self.test.df])

    @property
    def target(self):
        if self.train.target is None or self.test.target is None:
            return None
        return np.concatenate([self.train.target, self.test.target])

    @classmethod
    def from_csv(cls, filepath):
        return cls(pd.read_csv(filepath))

    @classmethod
    def load(cls, output_dir=PATH_DATA_DIR):
        return cls(DataBuilder.load(output_dir=output_dir))
