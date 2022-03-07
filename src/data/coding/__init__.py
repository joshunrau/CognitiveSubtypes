import json
import pkg_resources as pkg

def load_json(resource_name):
    filepath = pkg.resource_filename(__name__, resource_name)
    with open(filepath) as file:
        return json.load(file)

MEDICATION_CODES = load_json("medications.json")
