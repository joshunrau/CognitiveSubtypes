import pandas as pd

from .base import BaseData

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Dataset:
    
    def __init__(self):

        self.data = BaseData.load()

        self.cognitive_feature_names = [
            "meanReactionTimeTest",
            "timeTrailMakingTestA",
            "timeTrailMakingTestB",
            "accuracyTowerTest", 
            "accuracySymbolDigitTest",
            "incorrectPairsMatchingTask",
            "prospectiveMemoryTask"
        ]
    
        self.imaging_feature_names = []
        for col in self.data.columns:
            if col.startswith("area") or col.startswith("thickness") or col.startswith("volume"):
                self.imaging_feature_names.append(col)
    
        self.feature_names = self.cognitive_feature_names + self.imaging_feature_names

        self.scaler = StandardScaler()
        training_data, testing_data = train_test_split(self.data, test_size=0.33, random_state=42)
        training_data[self.feature_names] = self.scaler.fit_transform(training_data[self.feature_names])
        testing_data[self.feature_names] = self.scaler.transform(testing_data[self.feature_names])
        self.train, self.test = Features(self, training_data), Features(self, testing_data)
    
class Features:

    def __init__(self, obj: Dataset, data: pd.DataFrame) -> None:
        if not isinstance(obj, Dataset) or not isinstance(data, pd.DataFrame):
            raise TypeError
        self.data = data
        self.cognitive = data[obj.cognitive_feature_names].to_numpy()
        self.imaging = data[obj.imaging_feature_names].to_numpy()

    def __str__(self) -> str:
        return "\n".join([
            "Cognitive Feature Set: " + str(self.cognitive.shape),
            "Imaging Feature Set: " + str(self.imaging.shape)
        ])

        