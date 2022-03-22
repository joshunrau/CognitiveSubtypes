from data.build import BiobankData
from data.dataset import Dataset
from models.classify import BestSVC, BestRandomForestClassifier, BestRidgeClassifier
from models.cluster import BestKMeans


CURRENT_CSV="/Users/joshua/Developer/CognitiveSubtypes/data/raw/current.csv"

def main():

    ukbb_data = BiobankData(CURRENT_CSV, rm_na=True, subset_dx=True)
    data = Dataset(ukbb_data.df)

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
            print("Found new best classifier: " + str(model))
            print("Score: " + str(score))
    return best_classifier

if __name__ == "__main__":
    main()
    
    

