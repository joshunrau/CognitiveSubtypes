import numpy as np
import pandas as pd

from .base import BaseData
from ..visualization.plots import plot_distributions

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, PowerTransformer


class Dataset:
    
    def __init__(self):

        self.df = BaseData.load()

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
        for col in self.df.columns:
            if col.startswith("area") or col.startswith("thickness") or col.startswith("volume"):
                self.imaging_feature_names.append(col)
        
        self.feature_names = self.cognitive_feature_names + self.imaging_feature_names
        
        log_transformer = make_pipeline(FunctionTransformer(np.log), StandardScaler())
        power_transformer =  PowerTransformer()

        self.transforms = [
            [log_transformer, ["meanReactionTimeTest", "timeTrailMakingTestA", "timeTrailMakingTestB"]],
            [power_transformer, ["accuracyTowerTest", "accuracySymbolDigitTest", "incorrectPairsMatchingTask"]],
            [StandardScaler(), ["prospectiveMemoryTask"]]
        ]

        training_data, testing_data = train_test_split(self.df, test_size=0.33, random_state=42)

        for transformer, variables in self.transforms:
            training_data[variables] = transformer.fit_transform(training_data[variables])
            testing_data[variables] = transformer.transform(testing_data[variables])
            self.df[variables] = transformer.transform(self.df[variables])

        self.train, self.test = Features(self, training_data), Features(self, testing_data)
    
    def plot_cognitive_features(self, filepath=None):
        plot_distributions(self.df, self.cognitive_feature_names, filepath=filepath)

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

        