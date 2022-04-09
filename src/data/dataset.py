import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.utils.validation import check_is_fitted

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
    def demo_feature_names(self):
        return ['age']
    
    @property
    def cognitive_feature_names(self):
        return [
            "meanReactionTimeTest",
            "timeTrailMakingTestA",
            "timeTrailMakingTestB",
            "correctTowerTest",
            "correctSymbolDigitTest",
            "incorrectPairsMatchingTask",
            "prospectiveMemoryTask",
            'maxDigitsNumericMemoryTest'
        ]
    
    def feature_names_startswith(self, s: str):
        return [x for x in self.df.columns if x.startswith(s)]
    
    @property
    def area_feature_names(self):
        return self.feature_names_startswith('area')
    
    @property
    def thickness_feature_names(self):
        return self.feature_names_startswith('thickness')

    @property
    def volume_feature_names(self):
        return self.feature_names_startswith('volume')
    
    @property
    def imaging_feature_names(self):
        return self.area_feature_names + self.thickness_feature_names + self.volume_feature_names

    @property
    def feature_names(self):
        return self.demo_feature_names + self.cognitive_feature_names + self.imaging_feature_names

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
    
    features_to_transform = [
        "meanReactionTimeTest",
        "timeTrailMakingTestA",
        "timeTrailMakingTestB",
        "correctSymbolDigitTest",
        "incorrectPairsMatchingTask",
    ]

    def __init__(self, df):
        train_data, test_data = train_test_split(df, test_size=0.25, random_state=0)
        self.train, self.test = Data(train_data), Data(test_data)

    def apply_transformer(self, transformer, vars_to_transform):
        check_is_fitted(transformer)
        self.train.df[vars_to_transform] = transformer.transform(self.train.df[vars_to_transform])
        self.test.df[vars_to_transform] = transformer.transform(self.test.df[vars_to_transform])

    def apply_scaler(self, scaler, vars_to_scale):
        check_is_fitted(scaler)
        self.train.df[vars_to_scale] = scaler.transform(self.train.df[vars_to_scale])
        self.test.df[vars_to_scale] = scaler.transform(self.test.df[vars_to_scale])
    
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
    
    @classmethod
    def load_patients(cls):
        data = DataBuilder.load()
        return cls(data[data['subjectType'] == 'patient'])
    
    @classmethod
    def load_controls(cls):
        data = DataBuilder.load()
        return cls(data[data['subjectType'] == 'control'])
    
    @classmethod
    def load_preprocess(cls):
        data = cls.load()
        scaler = StandardScaler()
        scaler.fit(data.train.df[data.cognitive_feature_names])
        data.apply_scaler(scaler, data.cognitive_feature_names)
        transformer = PowerTransformer(method='yeo-johnson')
        transformer.fit(data.train.df[cls.features_to_transform])
        data.apply_transformer(transformer, cls.features_to_transform)
        return data
        
    @classmethod
    def get_sets(cls):
        
        controls = Dataset.load_controls()
        patients = Dataset.load_patients()
        
        transformer = PowerTransformer(method='yeo-johnson')
        transformer.fit(patients.df[cls.features_to_transform])
        patients.apply_transformer(transformer, cls.features_to_transform)
        transformer.fit(controls.df[cls.features_to_transform])
        controls.apply_transformer(transformer, cls.features_to_transform)
        transformer.fit(controls.df[cls.features_to_transform])
        
        scaler = StandardScaler()
        scaler.fit(controls.df[controls.cognitive_feature_names])
        controls.apply_scaler(scaler, controls.cognitive_feature_names)
        patients.apply_scaler(scaler, patients.cognitive_feature_names)

        return patients, controls
