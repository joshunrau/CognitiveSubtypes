from .base import Features

class ImagingFeatures(Features):
    
    def __init__(self):
        super().__init__()
    
    @property
    def names(self):
        return [s for s in self.df.columns if s.startswith("volume")]
    
