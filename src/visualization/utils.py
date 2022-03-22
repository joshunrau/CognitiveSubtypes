import re

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