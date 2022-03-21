import os
import re

from datetime import date, datetime

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, StandardScaler

from . import DATA_DIR
from .variables import load_variables


class RawData:
    """ class to subset and recode raw data from the ukbb """

    path_tabular_data = os.path.join(DATA_DIR, "raw", "current.csv")
    
    if not os.path.isfile(path_tabular_data):
        raise FileNotFoundError
    
    idvar = "id"
    variables = load_variables()

    included_diagnoses = {
        "anySSD": "F2\d",
        "anyMoodDisorder": "F3\d"
    }

    excluded_diagnoses = {
        "anyDementia": "F0\d"
    }

    edu_levels = {
        "EduNoneOfTheAbove": "None of the above",
        "EduDeclineToAnswer": "Prefer not to answer",
        "EduUniversityDegree": "College or University degree",
        "EduALevelsOrEq": "A levels/AS levels or equivalent",
        "EduOLevelsOrEq": "O levels/GCSEs or equivalent",
        "EduCSEOrEq": "CSEs or equivalent",
        "EduNVQOrEq": "NVQ or HND or HNC or equivalent",
        "EduOtherProfQual": "Other professional qualifications eg: nursing, teaching"
    }
    
    selected_diagnoses = included_diagnoses | excluded_diagnoses
    
    def __init__(self, subset_dx=True, rm_na=False) -> None:
        
        # Import, recode, and subset tabular data
        ukbb_vars, recoded_vars = self.get_var_names()

        self.df = pd.read_csv(self.path_tabular_data, dtype=str, usecols=ukbb_vars)
        self.df.rename({k: v for k, v in zip(ukbb_vars, recoded_vars)}, axis=1, inplace=True)
        self.df.dropna(axis=1, how="all", inplace=True)

        # Recode array vars
        self.df = self.add_binary_variables(self.df, "educationalQualifications", self.edu_levels, drop_target=True)
        self.df = self.add_binary_variables(self.df, "diagnoses", self.selected_diagnoses, drop_target=True)
        
        # Apply inclusion/exclusion criteria for diagnoses
        if subset_dx:
            list_series = [self.df[key] == True for key in self.included_diagnoses]
            self.df = self.df[pd.concat(list_series, axis=1).any(axis=1)]
            for key in self.excluded_diagnoses:
                self.df = self.df[self.df[key] == False]

        # Recode variable values
        for name in self.variables:
            cols = [col for col in self.df.columns if col.startswith(name)]
            if self.variables[name]["Included"] and cols != [] and self.variables[name]["Coding"] is not None:
                self.df[cols] = self.df[cols].replace(to_replace=self.variables[name]["Coding"])

        self.df.reset_index(drop=True, inplace=True)
        self.df = self.df.apply(pd.to_numeric, errors="ignore")

        if rm_na:
            self.df.dropna(how="any", inplace=True)
        
        # Merge incorrectPairsMatchingTask
        self.df["incorrectPairsMatchingTask"] = self.df["incorrectPairsMatchingTask1"] + self.df["incorrectPairsMatchingTask2"] + self.df["incorrectPairsMatchingTask3"]
        self.df.drop(["incorrectPairsMatchingTask1", "incorrectPairsMatchingTask2", "incorrectPairsMatchingTask3"], axis=1, inplace=True)
        
        # Compute accuracy variables
        self.df["accuracyTowerTest"] = self.df["correctTowerTest"] / self.df["attemptsTowerTest"]
        self.df["accuracySymbolDigitTest"] = self.df["correctSymbolDigitTest"] / self.df["attemptsSymbolDigitTest"]

    def get_var_names(self) -> tuple:
        """ Return lists of actual and recoded variable names based on config """
        ukbb_vars, recoded_vars = ["eid"], [self.idvar]
        for var in self.variables:
            if self.variables[var]["Included"]:
                array_vars = []
                for i in self.variables[var]['ArrayRange']:
                    array_vars.append(f"{self.variables[var]['DataField']}-{self.variables[var]['InstanceNum']}.{i}")
                ukbb_vars += array_vars
                if len(self.variables[var]['ArrayRange']) == 1:
                    recoded_vars.append(var)
                else:
                    array_vars = []
                    for i in self.variables[var]['ArrayRange']:
                        array_vars.append(f"{var}{i}")
                    recoded_vars += array_vars
        if len(ukbb_vars) != len(recoded_vars):
            raise ValueError
        return ukbb_vars, recoded_vars
    
    def add_binary_variables(self, df: pd.DataFrame, target: str, patterns: dict, drop_target=False):
        """ 
        Takes as input a variable of interest and a dictionary with keys representing new
        variable names mapped onto regular expressions. New binary variables will be created
        based on whether each individual has a value matching the regular expression in any 
        of the columns related to the variable of interest.
        """

        cols = [col for col in df if col.startswith(target)]
        all_vars = list(patterns.keys())
        new_vars = {var_name: [] for var_name in [self.idvar] + all_vars}

        for index, row in df[cols].iterrows():
            new_vars[self.idvar].append(df[self.idvar][index])
            for pat in patterns:
                for value in row:
                    try:
                        if re.match(patterns[pat], value) is not None:
                            new_vars[pat].append(True)
                            break
                    except TypeError:
                        continue
                if len(new_vars[self.idvar]) != len(new_vars[pat]):
                    new_vars[pat].append(False)

        if not sum([len(x) for x in new_vars.values()]) == len(new_vars[self.idvar]) * len(new_vars.keys()):
            raise ValueError(f"{sum([len(x) for x in new_vars.values()])} != {len(new_vars['eid']) * len(new_vars.keys())}")
        
        new_df = pd.DataFrame(new_vars)
        if drop_target:
            df.drop(cols, axis=1, inplace=True)
        return pd.merge(df, new_df, left_on=self.idvar, right_on=self.idvar)

    def write_csv(self):
        filename = os.path.join(DATA_DIR, "processed", f"dataset_{date.today().isoformat()}.csv")
        self.df.to_csv(filename, index=False)

    @classmethod
    def get_latest_filepath(cls):
        """ Return the path to most recent saved dataset """

        processed_dir = os.path.join(DATA_DIR, "processed")
        newest_date, newest_file = None, None

        for filename in os.listdir(processed_dir):
            try:
                file_date = datetime.fromisoformat(filename.strip("dataset_").strip(".csv"))
            except ValueError:
                continue
            if newest_date is None or file_date > newest_date:
                newest_date, newest_file = file_date, filename

        if newest_date is None:
            return None

        return os.path.join(processed_dir, newest_file)

    @classmethod
    def load(cls):
        """ Returns the most recent saved dataframe """
        
        filepath = cls.get_latest_filepath()
        if filepath is None:
            raise FileNotFoundError("Could not find existing dataset")
        
        return pd.read_csv(filepath)


class SubsetData:
    
    def __init__(self, df):
        self._df = df
        self._target = None
    
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
    

class Dataset(SubsetData):
    
    def __init__(self, df=RawData.load()):
        train_data, test_data = train_test_split(df, test_size=0.33, random_state=42)
        self.train, self.test = SubsetData(train_data), SubsetData(test_data)
    
    def apply_transforms(self):

        log_transformer = FunctionTransformer(np.log)
        power_transformer =  PowerTransformer(standardize=False)

        self.transforms = [
            [log_transformer, ["meanReactionTimeTest", "timeTrailMakingTestA", "timeTrailMakingTestB"]],
            [power_transformer, ["accuracyTowerTest", "accuracySymbolDigitTest", "incorrectPairsMatchingTask"]]
        ]

        for transformer, variables in self.transforms:
            self.train.df[variables] = transformer.fit_transform(self.train.df[variables])
            self.test.df[variables] = transformer.transform(self.test.df[variables])
        
    def apply_scaler(self, scaler = StandardScaler()):
        self.train.df[self.train.feature_names] = scaler.fit_transform(self.train.features)
        self.test.df[self.test.feature_names] = scaler.transform(self.test.features)
    
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
    def from_csv(cls, filepath, **kwargs):
        return cls(pd.read_csv(filepath), **kwargs)