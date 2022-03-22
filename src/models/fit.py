from ..data.dataset import Dataset
from .cluster import BestKMeans
from .classify import BestSVC, BestRandomForestClassifier, BestRidgeClassifier

def fit_model(data: Dataset):

    model = BestKMeans()
    model.fit(data)
    data.train.target, data.test.target = model.predict(data)
    
    classifiers = [
        BestSVC(),
        BestRandomForestClassifier(),
        BestRidgeClassifier()
    ]
    
    best_score = -1
    best_classifier = None
    
    for model in classifiers:
        model.fit(data)
        score = model.score(data)
        if score > best_score:
            best_score = score
            best_classifier = model
    
    return best_classifier