import json
import re

import numpy as np
import pandas as pd


def ukbb_tsv_to_json(path_tsv: str, path_json: str) -> None:
    df = pd.read_csv(path_tsv, sep="\t")
    contents = {}
    for _, row in df.iterrows():
        contents[row["coding"]] = row["meaning"]
    with open(path_json, "w") as file:
        file.write(json.dumps(contents, indent=4))


def is_instantiated(x):
    try:
        issubclass(x, object)
    except TypeError:
        return True
    return False


def get_array_counts(arr: np.array) -> pd.DataFrame:
    values, counts = np.unique(arr, return_counts=True)
    assert len(values) == len(counts)

    value_counts = {}
    for i, k in enumerate(values):
        value_counts[k] = {
            "Count": counts[i],
            "Percent": round(counts[i] / sum(counts), 2) * 100
        }

    return pd.DataFrame.from_dict(value_counts, orient='index')


def apply_dict_keys(d, f):
    return {f(k): v for k, v in d.items()}


def camel_case_split(identifier: str) -> str:
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return " ".join([m.group(0).capitalize() for m in matches])


def is_number(x: str) -> bool:
    try:
        int(x)
    except ValueError:
        return False
    return True


def flatten_list(list_of_lists: list) -> list:
    """ flatten the first dimension of a list of lists """
    flattened_list = []
    for sublist in list_of_lists:
        if not isinstance(sublist, list):
            raise TypeError(f"Expected list, not {type(sublist)}")
        for item in sublist:
            flattened_list.append(item)
    return flattened_list
