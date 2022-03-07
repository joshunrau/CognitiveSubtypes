import os

from abc import ABC

from .coding import DIAGNOSTIC_CODES, MEDICATION_CODES


class Config(ABC):
    """
    
    This class is used to encapsulate all data that can easily be manually adjusted by
    each of us. It is inherited by the Dataset class and there is no reason for it to 
    ever be instantiated.

    Attributes
    ----------
    
    data_dir: str
        Path to the directory containing any data files and where the subsetted dataset will be outputted
    
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
    path_tabular_data = os.path.join(data_dir, "raw", "tabular.csv")

    if not os.path.isdir(data_dir):
        raise NotADirectoryError
    
    if not os.path.isfile(path_tabular_data):
        raise FileNotFoundError
    
    idvar = "id"

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
            'Coding': DIAGNOSTIC_CODES
        },
        'handedness': {
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
        'age': {
            'DataField': 21003,
            'InstanceNum': 2,
            'ArrayRange': range(0, 1),
            'Included': True,
            'Coding': None
        },
        'meanReactionTimeTest': {
            'DataField': 20023,
            'InstanceNum': 2,
            'ArrayRange': range(0, 1),
            'Included': True,
            'Coding': None
        },
        'maxDigitsNumericMemoryTest': {
            'DataField': 4282,
            'InstanceNum': 2,
            'ArrayRange': range(0, 1),
            'Included': True,
            'Coding': None
        },
        'timeTrailMakingTestA': {
            'DataField': 6348,
            'InstanceNum': 2,
            'ArrayRange': range(0, 1),
            'Included': True,
            'Coding': None
        },
        'timeTrailMakingTestB': {
            'DataField': 6350,
            'InstanceNum': 2,
            'ArrayRange': range(0, 1),
            'Included': True,
            'Coding': None
        },
        'attemptsTowerTest': {
            'DataField': 6383,
            'InstanceNum': 2,
            'ArrayRange': range(0, 1),
            'Included': True,
            'Coding': None
        },
        'correctTowerTest': {
            'DataField': 21004,
            'InstanceNum': 2,
            'ArrayRange': range(0, 1),
            'Included': True,
            'Coding': None
        },
        'attemptsSymbolDigitTest': {
            'DataField': 23323,
            'InstanceNum': 2,
            'ArrayRange': range(0, 1),
            'Included': True,
            'Coding': None
        },
        'correctSymbolDigitTest': {
            'DataField': 23324,
            'InstanceNum': 2,
            'ArrayRange': range(0, 1),
            'Included': True,
            'Coding': None
        },
        'resultProspectiveMemoryTask': {
            'DataField': 20018,
            'InstanceNum': 2,
            'ArrayRange': range(0, 1),
            'Included': True,
            'Coding': None
        },
        'incorrectPairsMatchingTask': {
            'DataField': 399,
            'InstanceNum': 2,
            'ArrayRange': range(1, 4),
            'Included': True,
            'Coding': None
        },
        'correctAssociateLearningTest': { # Begin not included
            'DataField': 20197,
            'InstanceNum': 2,
            'ArrayRange': range(0, 1),
            'Included': False, 
            'Coding': None
        },
        'educationalQualifications': {
            'DataField': 6138,
            'InstanceNum': 2,
            'ArrayRange': range(0, 6),
            'Included': False,
            'Coding': None
        },
        'medications': {
            'DataField': 20003,
            'InstanceNum': 2,
            'ArrayRange': range(0, 48),
            'Included': False,
            'Coding': MEDICATION_CODES
        },
    }

    included_diagnoses = {
        "anySSD": "F2\d",
        "anyMoodDisorder": "F3\d"
    }

    excluded_diagnoses = {
        "anyDementia": "F0\d"
    }

    selected_diagnoses = included_diagnoses | excluded_diagnoses