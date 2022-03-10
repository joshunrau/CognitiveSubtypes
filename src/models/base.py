from abc import ABC, abstractmethod

class Model(ABC):
    
    def __init__(self) -> None:
        pass
    
    @property
    @abstractmethod
    def features(self):
        pass
    
    @property
    @abstractmethod
    def pipeline(self):
        pass