from pkg_resources import resource_filename
from ...utils import load_json

DIAGNOSTIC_CODES = load_json(resource_filename(__name__, "diagnoses.json"))
MEDICATION_CODES = load_json(resource_filename(__name__, "medications.json"))