import re

def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return " ".join([m.group(0).capitalize() for m in matches])

def is_number(x: str):
    try:
        int(x)
    except ValueError:
        return False
    return True

    
