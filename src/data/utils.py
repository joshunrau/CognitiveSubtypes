import json

def parse_range(dct):
    if "ArrayRange" in dct:
        ar = dct["ArrayRange"]
        dct["ArrayRange"] = range(ar["start"], ar["stop"], ar["step"])
    return dct

def load_json(filepath):
    with open(filepath) as file:
        return json.loads(file.read(), object_hook=parse_range)