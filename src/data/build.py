import copy
import os
import re
from datetime import date, datetime

import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from .variables import load_variables
from ..filepaths import PATH_CURRENT_CSV, PATH_DATA_DIR


class DataBuilder:
    """ class to subset and recode raw data """

    idvar = "id"
    variables = load_variables()
    
    included_diagnoses = {
        "anySSD": r"F2\d",
        "anyMoodDisorder": r"F3\d"
    }

    excluded_diagnoses = {
        "anyDementia": r"F0\d"
    }

    any_mental_disorder = {
        "anyMentalDisorder": r"F\d*"
    }

    selected_diagnoses = included_diagnoses | excluded_diagnoses | any_mental_disorder

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

    def __init__(self, drop_dx: bool = True, drop_na: bool = True, path_csv: str = PATH_CURRENT_CSV, verbose: bool = False) -> None:
        
        self._verbose = verbose

        self.printv("Begin getting variable names...")
        ukbb_vars, recoded_vars = self.get_var_names()
        self.printv("Done!")

        self.printv("Begin reading CSV...")
        self.df = pd.read_csv(path_csv, dtype=str, usecols=ukbb_vars)
        self.df.rename({k: v for k, v in zip(ukbb_vars, recoded_vars)}, axis=1, inplace=True)
        self.df.dropna(axis=1, how="all", inplace=True)
        self.printv("Done!")

        self.printv("Begin getting education...")
        self.df = self.add_binary_variables(self.df, "educationalQualifications", self.edu_levels, drop_target=True)
        self.printv("Done!")

        self.printv("Begin adding diagnoses...")
        self.df = self.add_binary_variables(self.df, "diagnoses", self.selected_diagnoses, drop_target=drop_dx)
        self.printv("Done!")

        self.printv("Begin iterating...")
        for name in self.variables:
            self.printv(f"Variable: {name}")
            cols = [col for col in self.df.columns if col.startswith(name)]
            self.printv(f"Columns: {cols}")
            if self.variables[name]["Included"] and cols != [] and self.variables[name]["Coding"] is not None:
                self.printv(f"Replace: {self.variables[name]['Coding']}")
                self.df[cols] = self.df[cols].replace(to_replace=self.variables[name]["Coding"])

        self.df.reset_index(drop=True, inplace=True)
        self.df = self.df.apply(pd.to_numeric, errors="ignore")

        if drop_na:
            self.df.dropna(how="any", inplace=True)

        self.printv("Assigning diagnoses...")
        self.df = self.df.assign(dx=self.df.apply(self.compute_dx, axis=1))
        self.printv("Done!")

        self.df.drop(list(self.selected_diagnoses.keys()), axis=1)

        for key in self.excluded_diagnoses:
            self.df = self.df[self.df[key] == False]
        self.df = self.df[self.df['handedness'] == "Right-handed"]
        
        self.patient_df = self.get_patients()
        self.control_df = self.get_controls()
        self.matched_controls = self.get_matched_controls(self.patient_df, self.control_df)
        self.df = pd.concat([self.patient_df, self.matched_controls])
        assert sum(self.df['subjectType'] == 'patient') == sum(self.df['subjectType'] == 'control')
    
    def printv(self, msg: str):
        if self._verbose:
            print(msg)

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

    def add_binary_variables(self, df: pd.DataFrame, target: str, patterns: dict, drop_target: bool = False):
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
            raise ValueError(
                f"{sum([len(x) for x in new_vars.values()])} != {len(new_vars['eid']) * len(new_vars.keys())}")

        new_df = pd.DataFrame(new_vars)
        if drop_target:
            self.printv("Dropping target...")
            df.drop(cols, axis=1, inplace=True)
            self.printv("Finished dropping target!")
        return pd.merge(df, new_df, left_on=self.idvar, right_on=self.idvar)
    
    @staticmethod
    def compute_dx(df: pd.DataFrame):
        if df["anySSD"] & df["anyMoodDisorder"]:
            return "SSD + Mood Disorder"
        elif df["anySSD"]:
            return "Only SSD"
        elif df["anyMoodDisorder"]:
            return "Only Mood Disorder"
        elif ~df['anyDementia'] & ~df['anyMentalDisorder']:
            return None
        else:
            raise AssertionError

    def get_patients(self):
        list_series = [self.df[key] == True for key in self.included_diagnoses]
        df = copy.deepcopy(self.df[pd.concat(list_series, axis=1).any(axis=1)])
        df['subjectType'] = 'patient'
        return df

    def get_controls(self):
        list_series = [self.df[key] == False for key in self.any_mental_disorder]
        df = copy.deepcopy(self.df[pd.concat(list_series, axis=1).any(axis=1)])
        df['subjectType'] = 'control'
        return df
    
    def write_csv(self, output_dir: str = PATH_DATA_DIR):
        filename = os.path.join(output_dir, f"dataset_{date.today().isoformat()}.csv")
        self.df.to_csv(filename, index=False)
        
    @classmethod
    def get_latest_filepath(cls, output_dir: str = PATH_DATA_DIR):
        """ Return the path to most recent saved dataset """

        newest_date, newest_file = None, None
        for filename in os.listdir(output_dir):
            try:
                file_date = datetime.fromisoformat(filename.strip("dataset_").strip(".csv"))
            except ValueError:
                continue
            if newest_date is None or file_date > newest_date:
                newest_date, newest_file = file_date, filename

        if newest_date is None:
            return None

        return os.path.join(output_dir, newest_file)
    
    @classmethod
    def load(cls, output_dir: str = PATH_DATA_DIR):
        """ Returns the most recent saved dataframe """

        filepath = cls.get_latest_filepath(output_dir)
        if filepath is None:
            raise FileNotFoundError("Could not find existing dataset")

        return pd.read_csv(filepath)
    
    @staticmethod
    def get_matched_controls(patient_df: pd.DataFrame, control_df: pd.DataFrame):
        match_vars = ["age", "sex"]
        patient_features = patient_df[match_vars].to_numpy()
        patient_features[:, 1] = np.where(patient_features[:, 1] == "Male", 1, 0)
        control_features = control_df[match_vars].to_numpy()
        control_features[:, 1] = np.where(control_features[:, 1] == "Male", 1, 0)
        model = NearestNeighbors(algorithm='kd_tree')
        model.fit(control_features)
        _, neigh_ind = model.kneighbors(patient_features, n_neighbors=1)
        return control_df.iloc[neigh_ind.flatten()]