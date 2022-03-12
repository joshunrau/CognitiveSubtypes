import os
import re

from datetime import date, datetime

import pandas as pd

from .variables import load_variables

class Dataset:
    
    data_dir = "/Users/joshua/Developer/CognitiveSubtypes/data"
    path_tabular_data = os.path.join(data_dir, "raw", "current.csv")
    
    if not os.path.isdir(data_dir):
        raise NotADirectoryError
    
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
    
    selected_diagnoses = included_diagnoses | excluded_diagnoses
    
    def __init__(self, df) -> None:
        self.df = df
    
    def write_csv(self):
        filename = os.path.join(self.data_dir, "processed", f"dataset_{date.today().isoformat()}.csv")
        self.df.to_csv(filename, index=False)

    @classmethod
    def get_latest_filepath(cls):
        """ Return the path to most recent saved dataset """

        processed_dir = os.path.join(cls.data_dir, "processed")
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
    
    @classmethod
    def make(cls):
        """ Returns a Dataset instance using a dataframe created with the parameters specified in Config """

        def get_var_names() -> tuple:
            """ Return lists of actual and recoded variable names based on config """
            ukbb_vars, recoded_vars = ["eid"], [cls.idvar]
            for var in cls.variables:
                if cls.variables[var]["Included"]:
                    array_vars = []
                    for i in cls.variables[var]['ArrayRange']:
                        array_vars.append(f"{cls.variables[var]['DataField']}-{cls.variables[var]['InstanceNum']}.{i}")
                    ukbb_vars += array_vars
                    if len(cls.variables[var]['ArrayRange']) == 1:
                        recoded_vars.append(var)
                    else:
                        array_vars = []
                        for i in cls.variables[var]['ArrayRange']:
                            array_vars.append(f"{var}{i}")
                        recoded_vars += array_vars
            if len(ukbb_vars) != len(recoded_vars):
                raise ValueError
            return ukbb_vars, recoded_vars
        
        def add_binary_variables(df: pd.DataFrame, target: str, patterns: dict, drop_target=False):
            """ 
            Takes as input a variable of interest and a dictionary with keys representing new
            variable names mapped onto regular expressions. New binary variables will be created
            based on whether each individual has a value matching the regular expression in any 
            of the columns related to the variable of interest.
            """

            cols = [col for col in df if col.startswith(target)]
            all_vars = list(patterns.keys())
            new_vars = {var_name: [] for var_name in [cls.idvar] + all_vars}

            for index, row in df[cols].iterrows():
                new_vars[cls.idvar].append(df[cls.idvar][index])
                for pat in patterns:
                    for value in row:
                        try:
                            if re.match(patterns[pat], value) is not None:
                                new_vars[pat].append(True)
                                break
                        except TypeError:
                            continue
                    if len(new_vars[cls.idvar]) != len(new_vars[pat]):
                        new_vars[pat].append(False)

            if not sum([len(x) for x in new_vars.values()]) == len(new_vars[cls.idvar]) * len(new_vars.keys()):
                raise ValueError(f"{sum([len(x) for x in new_vars.values()])} != {len(new_vars['eid']) * len(new_vars.keys())}")
            
            new_df = pd.DataFrame(new_vars)
            if drop_target:
                df.drop(cols, axis=1, inplace=True)
            return pd.merge(df, new_df, left_on=cls.idvar, right_on=cls.idvar)
        
        # Import, recode, and subset tabular data
        ukbb_vars, recoded_vars = get_var_names()

        df = pd.read_csv(cls.path_tabular_data, dtype=str, usecols=ukbb_vars)
        df.rename({k: v for k, v in zip(ukbb_vars, recoded_vars)}, axis=1, inplace=True)
        df.dropna(axis=1, how="all", inplace=True)

        # Apply inclusion criteria for diagnoses
        df = add_binary_variables(df, "diagnoses", cls.selected_diagnoses, drop_target=True)
        list_series = [df[key] == True for key in cls.included_diagnoses]
        df = df[pd.concat(list_series, axis=1).any(axis=1)]

        # Apply exclusion criteria for diagnoses
        for key in cls.excluded_diagnoses:
            df = df[df[key] == False]

        # Recode variable values
        for name in cls.variables:
            cols = [col for col in df.columns if col.startswith(name)]
            if cls.variables[name]["Included"] and cols != [] and cls.variables[name]["Coding"] is not None:
                df[cols] = df[cols].replace(to_replace=cls.variables[name]["Coding"])

        df.reset_index(drop=True, inplace=True)
        df = df.apply(pd.to_numeric, errors="ignore")
        df.dropna(how="any", inplace=True)
        
        return cls(df)
    
    def summarize_na(self):
        return self.df.isnull().sum()