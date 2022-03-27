from pprint import pprint

import numpy as np

from src.data.dataset import Dataset
from src.models.cluster import BestKMeans
from src.models.search import ClassifierSearch
from src.visualization.figures import  DataTransformFigure, KMeansScoresFigure
from src.visualization.tables import ClusterTable, PatientsVsControlsTable

def main():

    np.random.seed(0)
    data = Dataset.load()
    data.apply_transforms()
    data.apply_scaler()

    figure1 = DataTransformFigure()
    figure1.plot(dpi=300)
    figure1.save()

    table1 = PatientsVsControlsTable(data)
    table1.save()

    clu = BestKMeans()
    clu.fit(data.cognitive)
    print(clu.predict(data.cognitive, k=3, return_counts=True))
    pprint(clu.scores)

    data.train.target = clu.predict(data.train.cognitive, k=3)
    data.test.target = clu.predict(data.test.cognitive, k=3)
    
    table2 = ClusterTable(data)
    table2.save()

    figure2 = KMeansScoresFigure(clu)
    figure2.plot(dpi=300)
    figure2.save()

    cs = ClassifierSearch(score_method='balanced_accuracy')
    cs.fit(data.train.imaging, data.train.target)
    cs.score(data.test.imaging, data.test.target)
    
    print(cs.results_)


if __name__ == "__main__":
    main()
