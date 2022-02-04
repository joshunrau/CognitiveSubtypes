import datetime
import os
from pathlib import Path


class Config:
    """
    
    This class is used to encapsulate all data that can easily be manually adjusted by
    each of us. It is inherited by the Dataset class and there is no reason for it to 
    ever be instantiated (if you don't know what this means, don't worry about it). 

    Attributes
    ----------
    
    name: str
        Your name
    
    data_dir: str
        Path to the directory containing any data files and where the subsetted dataset will be outputted
    
    filepaths: dict
        Specifies the paths to the raw data file and the output file to be created when 
        subsetting the dataset. By default, the output file is set to your name and the 
        current date to avoid confusion, but feel free to change this.

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

    name = "Josh"

    data_dir = os.path.join(Path.home(), "projects", "def-mlepage", "UKBB")

    filepaths = {
        "RawData": os.path.join(data_dir, "raw_all_civet.csv"),
        "Output": os.path.join(data_dir, f"{name}_{datetime.date.today().isoformat()}.csv")
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