import pandas as pd

from .base import Features

class CognitiveFeatures(Features):
    
    names = [
        "meanReactionTimeTest", 
        "timeTrailMakingTestA", 
        "timeTrailMakingTestB", 
        "attemptsTowerTest", 
        "correctTowerTest", 
        "attemptsSymbolDigitTest",
        "correctSymbolDigitTest", 
        "incorrectPairsMatchingTask1",
        "incorrectPairsMatchingTask2", 
        "incorrectPairsMatchingTask3",
        "prospectiveMemoryTask",
    ]

    def __init__(self):
        super().__init__()
        self.recode_prospective_memory_task()
    
    def recode_prospective_memory_task(self):
        dummies = pd.get_dummies(self.df["prospectiveMemoryTask"])
        self.df = pd.merge(self.df, dummies, left_index=True, right_index=True)
        self.names.remove("prospectiveMemoryTask")
        self.names.extend(dummies.columns.tolist())
    

