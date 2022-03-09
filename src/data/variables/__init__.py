from pkg_resources import resource_filename
from ..utils import load_json


VARIABLES = load_json(resource_filename(__name__, "characteristics.json")) | load_json(resource_filename(__name__, "cognition.json")) | load_json(resource_filename(__name__, "imaging.json"))