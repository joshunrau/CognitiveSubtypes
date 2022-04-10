from abc import ABC, abstractmethod
import re
import os

import numpy as np
import pandas as pd
import scipy.stats as stats

from ..data.build import DataBuilder
from ..data.dataset import Dataset
from ..filepaths import PATH_RESULTS_DIR
from ..utils import camel_case_split


class Table(ABC):
    
    def __init__(self, data) -> None:
        self.data = data
        self.results = None

    @abstractmethod
    def get(self):
        pass

    def save(self, filename):
        if self.results is None:
            raise ValueError
        self.results.to_csv(os.path.join(PATH_RESULTS_DIR, "tables", filename))


class KWTestsPvC(Table):

    def get(self):
        
        df = self.data.df
        df = pd.concat([df, pd.get_dummies(df['sex'])], axis=1)

        patients_df = df[df['subjectType'] == 'patient']
        controls_df = df[df['subjectType'] == 'control']
        results = {}
        
        for variable in ['age', 'Female'] + self.data.cognitive_feature_names:
            result = stats.kruskal(patients_df[variable], controls_df[variable])
            results[variable] = {
                "Controls M": round(controls_df[variable].mean(), 2),
                "Controls SD": round(controls_df[variable].std(), 2),
                "Patients M": round(patients_df[variable].mean(), 2),
                "Patients SD": round(patients_df[variable].std(), 2),
                "H": round(result.statistic, 2),
                "p": round(result.pvalue, 3)
            }
        self.results = pd.DataFrame(results).T
        self.results.index = self.results.index.map(camel_case_split)
        self.results['p'] = np.where(self.results['p'] < .001, "<.001", self.results['p'])


class KWTestsClusters(Table):

    def get(self):
        df = self.data.df
        df = pd.concat([df, pd.get_dummies(df['sex'])], axis=1)
        df = pd.concat([df, pd.get_dummies(df['dx'])], axis=1)
        
        df0 = df[df['class'] == 0]
        df1 = df[df['class'] == 1]

        results = {}
        
        results = {}
        for variable in ['age', "Female", "Only Mood Disorder", \
                        "SSD + Mood Disorder", 'Only SSD'] + self.data.cognitive_feature_names:
            result = stats.kruskal(df0[variable], df1[variable])
            results[variable] = {
                "Class 0 M": round(df0[variable].mean(), 2),
                "Class 0 SD": round(df0[variable].std(), 2),
                "Class 1 M": round(df1[variable].mean(), 2),
                "Class 1 SD": round(df1[variable].std(), 2),
                "H": round(result.statistic, 2),
                "p": round(result.pvalue, 3)
            }
        self.results = pd.DataFrame(results).T
        self.results.index = self.results.index.map(camel_case_split)
        self.results['p'] = np.where(self.results['p'] < .001, "<.001", self.results['p'])

class KWTestsDX(Table):

    def get(self):
        
        df = self.data.df
        
        mood = df[df['dx'] == 'Only Mood Disorder']
        both = df[df['dx'] == 'SSD + Mood Disorder']
        ssd = df[df['dx'] == 'Only SSD']

        results = {}
        for variable in self.data.cognitive_feature_names:
            result = stats.kruskal(mood[variable], both[variable], ssd[variable])
            results[variable] = {
                "Only Mood Disorder M": round(mood[variable].mean(), 2),
                "Only Mood Disorder SD": round(mood[variable].std(), 2),
                "SSD + Mood Disorder M": round(both[variable].mean(), 2),
                "SSD + Mood Disorder SD": round(both[variable].std(), 2),
                "Only SSD M": round(ssd[variable].mean(), 2),
                "Only SSD SD": round(ssd[variable].std(), 2),
                "H": round(result.statistic, 2),
                "p": round(result.pvalue, 3)
            }
        self.results = pd.DataFrame(results).T
        self.results.index = self.results.index.map(camel_case_split)
        self.results['p'] = np.where(self.results['p'] < .001, "<.001", self.results['p'])


class Diagnoses(Table):
    
    def __init__(self):
        self.results = None
    
    def get(self):
        processed = Dataset.load_patients()
        data = DataBuilder(drop_dx=False, drop_na=False)
        df = pd.DataFrame(data.df[data.df['id'].isin(processed.df['id'])], copy=True)
        dx_cols = [col for col in data.df.columns if col.startswith("diagnoses")]
        df[dx_cols] = df[dx_cols].applymap(lambda x: x if type(x) == str and "F" in x else np.NaN)
        diagnoses = {}
        for idx, row in df[dx_cols].iterrows():
            subj_id = df.loc[idx, 'id']
            for j in row:
                if isinstance(j, float):
                    continue
                if re.match(f"F2\d", j) or re.match(r"F3\d", j):
                    if j not in diagnoses:
                        diagnoses[j] = [subj_id]
                    else:
                        diagnoses[j].append(subj_id)
        
        n = {k:len(v) for k, v in diagnoses.items()}
        perc = {k : round(v / 680 * 100, 2) for k,v in n.items()}
        ages = {}
        female = {}
        
        for dx in diagnoses:
            dx_age = [int(df.loc[df['id'] == subj_id, 'age']) for subj_id in diagnoses[dx]]
            dx_female = int(sum(np.where([df.loc[df['id'] == subj_id, 'sex'] == "Female" for subj_id in diagnoses[dx]], 1, 0)))
            ages[dx] = round(np.mean(dx_age), 2)
            female[dx] = round(dx_female / len(diagnoses[dx]) * 100, 2)
        
        self.results = pd.DataFrame({"N": n, "Percent": perc, 'Age': ages, "Female": female}).sort_index()