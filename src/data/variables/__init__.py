import json
import warnings

from pkg_resources import resource_filename

from .coding import VARIABLE_CODES


def load_variables():

    def parse_obj(dct):
        if "ArrayRange" in dct:
            ar = dct["ArrayRange"]
            dct["ArrayRange"] = range(ar["start"], ar["stop"], ar["step"])
        if "Coding" in dct and dct["Coding"] is not None:
            try:
                dct["Coding"] = VARIABLE_CODES[str(dct["Coding"])]
            except KeyError as err:
                warnings.warn("Could not find file for Biobank coding scheme: " + str(err))
                dct["Coding"] = None
        return dct

    def load_json(filepath):
        with open(filepath) as file:
            return json.loads(file.read(), object_hook=parse_obj)

    characteristics = load_json(resource_filename(__name__, "characteristics.json"))
    cognition = load_json(resource_filename(__name__, "cognition.json"))
    imaging = load_json(resource_filename(__name__, "imaging.json"))

    return characteristics | cognition | imaging
