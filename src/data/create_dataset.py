import re
import pandas as pd

from config import Config


class Dataset(Config):
    """

    Attributes
    ----------

    ukbb_vars: list
        Variable names based on user selections as coded in the Biobank.
    
    recoded_vars: list
        Variable names based on user selections as will be recoded.
    
    df: DataFrame

    Methods
    -------

    recode_diagnoses()
        Creates new variables for groups of diagnoses included or excluded, based on
        whether one or more of such diagnoses is present.

    apply_inclusion_criteria(method: str)
        Apply inclusion criteria based on specified method. Available options are "AND" and "OR".
    
    apply_exclusion_criteria()

    recode_vars()
        Replace values for each variable as specified in the config class 
    
    write_csv()
        Write self.df to the filepath specified in the config class 

    """

    ukbb_vars, recoded_vars = ["eid"], ["eid"]
    for var in Config.variables:
        if Config.variables[var]["Included"]:
            array_vars = []
            for i in Config.variables[var]['ArrayRange']:
               array_vars.append(f"{Config.variables[var]['DataField']}-{Config.variables[var]['InstanceNum']}.{i}")
            ukbb_vars += array_vars
            if len(Config.variables[var]['ArrayRange']) == 1:
                recoded_vars.append(f"{var}_t{Config.variables[var]['InstanceNum']}")
            else:
                array_vars = []
                for i in Config.variables[var]['ArrayRange']:
                    array_vars.append(f"{var}_t{Config.variables[var]['InstanceNum']}_{i}")
                recoded_vars += array_vars
    assert len(ukbb_vars) == len(recoded_vars)

    def __init__(self) -> None:
        self.df = pd.read_csv(self.filepaths["RawData"], dtype=str, usecols=self.ukbb_vars)
        self.df.rename({k: v for k, v in zip(self.ukbb_vars, self.recoded_vars)}, axis=1, inplace=True)
        self.df.dropna(axis=1, how="all", inplace=True)

    def recode_diagnoses(self):
        dx_cols = [col for col in self.df if col.startswith("diagnoses")]
        all_dx = list(self.selected_diagnoses.keys())
        new_vars = {var_name: [] for var_name in ["eid"] + all_dx}

        for i in range(len(self.df)):
            new_vars["eid"].append(self.df["eid"][i])
            for col in dx_cols:
                value = self.df[col][i]
                if pd.isnull(value):
                    for dx in all_dx:
                        if len(new_vars[dx]) != len(new_vars["eid"]):
                            new_vars[dx].append(False)
                    break
                for dx in self.selected_diagnoses:
                    if re.match(self.selected_diagnoses[dx], value) is not None:
                        if len(new_vars[dx]) != len(new_vars["eid"]):
                            new_vars[dx].append(True)

        assert sum([len(x) for x in new_vars.values()]) == len(new_vars["eid"]) * len(new_vars.keys())

        new_df = pd.DataFrame(new_vars)
        self.df = pd.merge(self.df, new_df, left_on="eid", right_on="eid")
        self.df.drop(dx_cols, axis=1, inplace=True)

    def apply_inclusion_criteria(self, method: str):
        if method == "AND":
            for key in self.included_diagnoses:
                self.df = self.df[self.df[key] == True]
        elif method == "OR":
            list_series = [self.df[key] == True for key in self.included_diagnoses]
            included = pd.concat(list_series, axis=1).any(axis=1)
            self.df = self.df[included]
        else:
            raise ValueError("Available methods: 'AND', 'OR'")

    def apply_exclusion_criteria(self):
        for key in self.excluded_diagnoses:
            self.df = self.df[self.df[key] == False]

    def recode_vars(self):
        for name in self.variables:
            cols = [col for col in self.df.columns if col.startswith(name)]
            if self.variables[name]["Included"] and cols != []:
                self.df[cols] = self.df[cols].replace(to_replace=self.variables[name]["Coding"])

    def write_csv(self):
        self.df.to_csv(self.filepaths["Output"], index=False)


def main():
    data = Dataset()
    data.recode_diagnoses()
    data.apply_inclusion_criteria(method="OR")
    data.apply_exclusion_criteria()
    data.recode_vars()
    data.write_csv()


if __name__ == "__main__":
    main()
