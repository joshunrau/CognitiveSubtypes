import os

import numpy as np
import pandas as pd
import scipy.stats as stats

from ..data.dataset import Dataset
from ..models.cluster import BestKMeans
from ..filepaths import PATH_RESULTS_DIR
from ..utils import camel_case_split


def kruskal_pvc():
    
    data = Dataset.load_preprocess()
    df = data.df
    df = pd.concat([df, pd.get_dummies(df['sex'])], axis=1)

    patients_df = df[df['subjectType'] == 'patient']
    controls_df = df[df['subjectType'] == 'control']
    results = {}
    
    for variable in ['age', 'Female'] + data.cognitive_feature_names:
        result = stats.kruskal(patients_df[variable], controls_df[variable])
        results[variable] = {
            "Controls M": round(controls_df[variable].mean(), 2),
            "Controls SD": round(controls_df[variable].std(), 2),
            "Patients M": round(patients_df[variable].mean(), 2),
            "Patients SD": round(patients_df[variable].std(), 2),
            "H": round(result.statistic, 2),
            "p": round(result.pvalue, 3)
        }
    results = pd.DataFrame(results).T
    results.index = results.index.map(camel_case_split)
    results['p'] = np.where(results['p'] < .001, "<.001", results['p'])
    
    results.to_csv(os.path.join(PATH_RESULTS_DIR, "tables", "kruskal_pvc.csv"))


def kruskal_cls():
    
    data = Dataset.load_preprocess()
    clu = BestKMeans()
    clu.fit(data.cognitive)
    
    data.train.target = clu.predict(data.train.cognitive, k=2)
    data.test.target = clu.predict(data.test.cognitive, k=2)

    df = data.df
    df = pd.concat([df, pd.get_dummies(df['sex'])], axis=1)
    df = pd.concat([df, pd.get_dummies(df['dx'])], axis=1)
    
    df0 = df[df['class'] == 0]
    df1 = df[df['class'] == 1]

    results = {}
    
    results = {}
    for variable in ['age', "Female", "Only Mood Disorder", \
                     "SSD + Mood Disorder", 'Only SSD'] + data.cognitive_feature_names:
        result = stats.kruskal(df0[variable], df1[variable])
        results[variable] = {
            "Class 0 M": round(df0[variable].mean(), 2),
            "Class 0 SD": round(df0[variable].std(), 2),
            "Class 1 M": round(df1[variable].mean(), 2),
            "Class 1 SD": round(df1[variable].std(), 2),
            "H": round(result.statistic, 2),
            "p": round(result.pvalue, 3)
        }
    results = pd.DataFrame(results).T
    results.index = results.index.map(camel_case_split)
    results['p'] = np.where(results['p'] < .001, "<.001", results['p'])

    results.to_csv(os.path.join(PATH_RESULTS_DIR, "tables", "kruskal_cls.csv"))