from .base import Features

class CognitiveFeatures(Features):
    
    names = [  # MISSING: maxDigitsNumericMemoryTest
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
        "prospectiveMemoryTaskNotRecalled",
        "prospectiveMemoryTaskFirstAttempt",
        "prospectiveMemoryTaskSecondAttempt",
    ]

    def __init__(self):
        super().__init__()
