import json
import os

from pkg_resources import resource_filename

def _load():
    codes = {}
    for filename in os.listdir(os.path.abspath(os.path.dirname(__file__))):
        if filename.startswith("coding"):
            coding_id = filename.strip("coding.json")
            with open(resource_filename(__name__, filename)) as file:
                codes[coding_id] = json.loads(file.read())
    return codes

VARIABLE_CODES = _load()