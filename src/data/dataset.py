from datetime import date
from pathlib import Path
from sklearn.model_selection import train_test_split

from .tabular import TabularData

class Dataset:
    
    def __init__(self, df) -> None:
        self.df = df
        self.features = df.drop(["id"], axis=1).to_numpy()
        return None
        self.target = df["QC"].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.target, test_size=0.25, random_state=0)
        self.train, self.test = {}, {}
        self.train["Features"], self.test["Features"] = X_train, X_test
        self.train["Target"], self.test["Target"] = y_train, y_test
    
    def write_csv(self):
        filepath = Path.cwd().joinpath("data", "processed", self.get_name_today())
        Path.mkdir(filepath.parent, parents=True, exist_ok=True)
        self.df.to_csv(filepath, index=False)
    
    @classmethod
    def make(cls):
        path_tabular_data = Path.cwd().joinpath("data", "raw", "tabular.csv")
        tabular_data = TabularData(path_tabular_data)
        return cls(tabular_data.df)
    
    @staticmethod
    def get_name_today():
        return f"dataset_{date.today().isoformat()}.csv"