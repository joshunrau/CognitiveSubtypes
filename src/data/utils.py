import json

import pandas as pd

def ukbb_tsv_to_json(path_tsv: str, path_json: str) -> None:
    df = pd.read_csv(path_tsv, sep="\t")
    contents = {}
    for _, row in df.iterrows():
        contents[row["coding"]] = row["meaning"]
    with open(path_json, "w") as file:
        file.write(json.dumps(contents, indent=4))