import pandas as pd
import numpy as np

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