import datetime
import os
import re

import pandas as pd


class Config:
    """
    
    This class is used to encapsulate all data that can easily be manually adjusted by
    each of us. It is inherited by the Dataset class and there is no reason for it to 
    ever be instantiated (if you don't know what this means, don't worry about it). 

    Attributes
    ----------
    
    data_dir: str
        Path to the directory containing any data files and where the subsetted dataset will be outputted
    
    filepaths: dict
        Specifies the paths to the raw data file and the output file to be created when 
        subsetting the dataset. By default, the output file is set to the current date 
        to avoid confusion, but feel free to change this.

    variables: dict
        Specifies the parameters of variables that may be included in the dataset. Additional 
        variables may be added as key-value pairs, where each variable is an arbitrary name, 
        mapped onto the following parameters:

            DataField: The unique identifier in the Biobank for that variable

            InstanceNum: The desired instance for that variable

            ArrayRange: The array of values associated with a DataField at a particular instance

            Included: Specifies whether this variable should be included in the subset

            Coding: Specifies labels used to recode values in dataset. If values are to be recoded, 
            then this should be a dictionary of the format 'value: label', where both values must be 
            strings. Otherwise, if the variable should not be recoded, or if it is recoded using a special 
            method in the Dataset class, as is the case for diagnoses, it should be 'None'.

        Note that the above parameters are provided on the Biobank showcase website.
    
    included_diagnoses: dict
        Specifies the diagnoses to be included in the dataset, which are given arbitrary names and map
        onto regular expressions to match the ICD diagnostic codes specified in the Biobank.

    excluded_diagnoses: dict
        Specifies the diagnoses to be excluded in the dataset, which are given arbitrary names and map
        onto regular expressions to match the ICD diagnostic codes specified in the Biobank.

    selected_diagnoses: dict
        Combined dictionary of all included and excluded diagnoses

    Notes
    -----
    ICD diagnostic codes are specified in the Biobank using coding method 19
    For more details, see https://biobank.ndph.ox.ac.uk/showcase/coding.cgi?id=19

    """

    data_dir = "/Users/joshua/Developer/CognitiveSubtypes/data"

    filepaths = {
        "RawData": os.path.join(data_dir, "raw_all_civet.csv"),
        "Output": os.path.join(data_dir, f"processed_{datetime.date.today().isoformat()}.csv")
    }

    variables = {
        'sex': {
            'DataField': 31,
            'InstanceNum': 0,
            'ArrayRange': range(0, 1),
            'Included': True,
            'Coding': {
                "0": "Female",
                "1": "Male"
            },
        },
        'diagnoses': {
            'DataField': 41270,
            'InstanceNum': 0,
            'ArrayRange': range(0, 226),
            'Included': True,
            'Coding': None
        },
        'hand': {
            'DataField': 1707,
            'InstanceNum': 0,
            'ArrayRange': range(0, 1),
            'Included': True,
            'Coding': {
                "1": "Right",
                "2": "Left",
                "3": "Both",
                "-3": "Decline to Answer"
            }
        },
        'year_birth': {
            'DataField': 34,
            'InstanceNum': 0,
            'ArrayRange': range(0, 1),
            'Included': True,
            'Coding': None
        },
        'age_assess': {
            'DataField': 21003,
            'InstanceNum': 2,
            'ArrayRange': range(0, 1),
            'Included': True,
            'Coding': None
        },
        'medication': {
            'DataField': 20003,
            'InstanceNum': 2,
            'ArrayRange': range(0, 48),
            'Included': False,
            'Coding': None
        },
        'edu_qual': {
            'DataField': 6138,
            'InstanceNum': 2,
            'ArrayRange': range(0, 6),
            'Included': False,
            'Coding': None
        },
        'mean_reaction_time': {
            'DataField': 20023,
            'InstanceNum': 2,
            'ArrayRange': range(0, 1),
            'Included': True,
            'Coding': None
        },
        'num_mem_max': {
            'DataField': 4282,
            'InstanceNum': 2,
            'ArrayRange': range(0, 1),
            'Included': True,
            'Coding': None
        },
        'trail_numeric': {
            'DataField': 6348,
            'InstanceNum': 2,
            'ArrayRange': range(0, 1),
            'Included': True,
            'Coding': None
        },
        'trail_alpha_numeric': {
            'DataField': 6350,
            'InstanceNum': 2,
            'ArrayRange': range(0, 1),
            'Included': True,
            'Coding': None
        },
        'tower_attempted': {
            'DataField': 6383,
            'InstanceNum': 2,
            'ArrayRange': range(0, 1),
            'Included': True,
            'Coding': None
        },
        'tower_correct': {
            'DataField': 21004,
            'InstanceNum': 2,
            'ArrayRange': range(0, 1),
            'Included': True,
            'Coding': None
        },
        'digit_sub_attempted': {
            'DataField': 23323,
            'InstanceNum': 2,
            'ArrayRange': range(0, 1),
            'Included': True,
            'Coding': None
        },
        'digit_sub_correct': {
            'DataField': 23324,
            'InstanceNum': 2,
            'ArrayRange': range(0, 1),
            'Included': True,
            'Coding': None
        },
        'assoc_learn_correct': {
            'DataField': 20197,
            'InstanceNum': 2,
            'ArrayRange': range(0, 1),
            'Included': False,
            'Coding': None
        },
        'prosp_mem': {
            'DataField': 20018,
            'InstanceNum': 2,
            'ArrayRange': range(0, 1),
            'Included': True,
            'Coding': None
        },
        'pairsmatch_incorr': {
            'DataField': 399,
            'InstanceNum': 2,
            'ArrayRange': range(1, 4),
            'Included': True,
            'Coding': None
        },
    }

    included_diagnoses = {
        "SSD": "F2\d",
        "MoodDisorder": "F3\d"
    }

    excluded_diagnoses = {
        "Dementia": "F0\d"
    }

    selected_diagnoses = included_diagnoses | excluded_diagnoses


class Dataset(Config):
    """

    Attributes
    ----------

    ukbb_vars: list
        Variable names based on user selections as coded in the Biobank.
    
    recoded_vars: list
        Variable names based on user selections as will be recoded.
    
    df: DataFrame

    Methods
    -------

    recode_diagnoses()
        Creates new variables for groups of diagnoses included or excluded, based on
        whether one or more of such diagnoses is present.

    apply_inclusion_criteria(method: str)
        Apply inclusion criteria based on specified method. Available options are "AND" and "OR".
    
    apply_exclusion_criteria()

    recode_vars()
        Replace values for each variable as specified in the config class 
    
    write_csv()
        Write self.df to the filepath specified in the config class 

    """

    ukbb_vars, recoded_vars = ["eid"], ["eid"]
    for var in Config.variables:
        if Config.variables[var]["Included"]:
            array_vars = []
            for i in Config.variables[var]['ArrayRange']:
               array_vars.append(f"{Config.variables[var]['DataField']}-{Config.variables[var]['InstanceNum']}.{i}")
            ukbb_vars += array_vars
            if len(Config.variables[var]['ArrayRange']) == 1:
                recoded_vars.append(f"{var}_t{Config.variables[var]['InstanceNum']}")
            else:
                array_vars = []
                for i in Config.variables[var]['ArrayRange']:
                    array_vars.append(f"{var}_t{Config.variables[var]['InstanceNum']}_{i}")
                recoded_vars += array_vars
    assert len(ukbb_vars) == len(recoded_vars)

    def __init__(self) -> None:
        self.df = pd.read_csv(self.filepaths["RawData"], dtype=str, usecols=self.ukbb_vars)
        self.df.rename({k: v for k, v in zip(self.ukbb_vars, self.recoded_vars)}, axis=1, inplace=True)
        self.df.dropna(axis=1, how="all", inplace=True)

    def recode_diagnoses(self):
        dx_cols = [col for col in self.df if col.startswith("diagnoses")]
        all_dx = list(self.selected_diagnoses.keys())
        new_vars = {var_name: [] for var_name in ["eid"] + all_dx}

        for i in range(len(self.df)):
            new_vars["eid"].append(self.df["eid"][i])
            for col in dx_cols:
                value = self.df[col][i]
                if pd.isnull(value):
                    for dx in all_dx:
                        if len(new_vars[dx]) != len(new_vars["eid"]):
                            new_vars[dx].append(False)
                    break
                for dx in self.selected_diagnoses:
                    if re.match(self.selected_diagnoses[dx], value) is not None:
                        if len(new_vars[dx]) != len(new_vars["eid"]):
                            new_vars[dx].append(True)

        assert sum([len(x) for x in new_vars.values()]) == len(new_vars["eid"]) * len(new_vars.keys())

        new_df = pd.DataFrame(new_vars)
        self.df = pd.merge(self.df, new_df, left_on="eid", right_on="eid")
        self.df.drop(dx_cols, axis=1, inplace=True)

    def apply_inclusion_criteria(self, method: str):
        if method == "AND":
            for key in self.included_diagnoses:
                self.df = self.df[self.df[key] == True]
        elif method == "OR":
            list_series = [self.df[key] == True for key in self.included_diagnoses]
            included = pd.concat(list_series, axis=1).any(axis=1)
            self.df = self.df[included]
        else:
            raise ValueError("Available methods: 'AND', 'OR'")

    def apply_exclusion_criteria(self):
        for key in self.excluded_diagnoses:
            self.df = self.df[self.df[key] == False]

    def recode_vars(self):
        for name in self.variables:
            cols = [col for col in self.df.columns if col.startswith(name)]
            if self.variables[name]["Included"] and cols != []:
                self.df[cols] = self.df[cols].replace(to_replace=self.variables[name]["Coding"])

    def write_csv(self):
        self.df.to_csv(self.filepaths["Output"], index=False)


def main():
    data = Dataset()
    data.recode_diagnoses()
    data.apply_inclusion_criteria(method="OR")
    data.apply_exclusion_criteria()
    data.recode_vars()
    data.write_csv()


if __name__ == "__main__":
    main()
