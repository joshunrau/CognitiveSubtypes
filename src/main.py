import os
import pandas as pd

from data.build import BiobankData
from data.dataset import Dataset
from models.classify import BestSVC, BestRandomForestClassifier, BestRidgeClassifier
from models.cluster import BestKMeans

CURRENT_CSV = "/Users/joshua/Developer/CognitiveSubtypes/data/raw/current.csv"
RESULTS_DIR = "/Users/joshua/Developer/CognitiveSubtypes/results"

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
    
    results = {}
    n_fit, n_remain = 0, len(classifiers)
    for model in classifiers:
        model.fit(data)
        score = model.score(data)
        results[str(model)] = {
            model.score_func.__name__: score
        }
        print("Finished fitting model: " + str(model))
        print(f"{model.score_func.__name__}: {score}")
        n_fit += 1
        n_remain -= 1
        print(f"Models Fit: {n_fit}")
        print(f"Models Remaining: {n_remain}\n")
    results = pd.DataFrame.from_dict(results, orient='index')
    results.to_csv(os.path.join(RESULTS_DIR, "model_fitting.csv"))

if __name__ == "__main__":
    main()
    
    

