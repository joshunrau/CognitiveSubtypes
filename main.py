import numpy as np
import pandas as pd

from src.data.dataset import Dataset
from src.models.classify import *
from src.models.cluster import BestKMeans


def main():
    np.random.seed(0)
    data, _ = Dataset.get_sets()
    clu = BestKMeans()
    clu.fit(data.cognitive)
    data.train.target = clu.predict(data.train.cognitive, k=2)
    data.test.target = clu.predict(data.test.cognitive, k=2)
    print(clu.predict(data.cognitive, k=2, return_counts=True))
    classifiers = [clf(score_method='roc_auc') for clf in [BestRandomForestClassifier]]
    for clf in classifiers:
        clf.fit(data.train.imaging, data.train.target, verbose=True)

if __name__ == "__main__":
    main()
