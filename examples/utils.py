# coding: utf-8
import os

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import average_precision_score

from dmatch.utils import CSV_READ_FORMAT
from dmatch.utils import Bootstrap, Score


def get_path(index_dir, index_name):
    train = os.path.join(index_dir, index_name + "_train")
    test = os.path.join(index_dir, index_name + "_test")
    return train, test


def load_reference(refpath, mapper):
    df = pd.read_csv(
        refpath,
        **CSV_READ_FORMAT,
        index_col=[0,1]
    )
    for key, value in mapper.items():
        df.rename(lambda x: x.replace(key, value), axis='index', inplace=True)
    return df

def load_dataframe(indexpath, reference):
    scores = pd.read_csv(
        os.path.join(indexpath, 'scores.csv'),
        **CSV_READ_FORMAT,
        index_col=[0,1]
    )
    df = scores.merge(reference, on=['entityA', 'entityB'])
    return df

def load_data(index_dir, index_name, reference_name, mapper):
    train, test = get_path(index_dir, index_name)
    train_df = load_dataframe(train, load_reference(reference_name, mapper['train']))
    test_df = load_dataframe(test, load_reference(reference_name, mapper['test']))
    return  train_df, test_df
    

class Model:
    def __init__(self, BaseModel, params):
        self.BaseModel = BaseModel
        self.params = params
        self.model = None
        self.best_model = None
        self.best_params = None
        self.scores = None
    
    def train(self, data):
        X = data.drop('Y', axis=1)
        y = data.Y
        model = GridSearchCV(self.BaseModel, self.params, scoring='f1', cv=StratifiedKFold(n_splits=5), verbose=1)
        model.fit(X, y)
        self.model = model
        self.best_model = model.best_estimator_
        self.best_params = model.best_params_
        return self.model
    
    def test(self, data, rep=1000):
        self.scores = Bootstrap.score(self.best_model, data, rep=rep, rate=1)
        df = data.copy()
        df['y_proba'] = self.best_model.predict_proba(df.drop('Y', axis=1))[:,1]
        self.scores['average_precision_score'] = average_precision_score(data.Y, self.best_model.predict_proba(data.drop('Y', axis=1))[:, 1])
        self.scores['tm_score_A_to_B'] = Score.tm_score(df, 'entityA')
        self.scores['tm_score_B_to_A'] = Score.tm_score(df, 'entityB')
        return self.scores

    def print_scores(self):
        for name, value in self.scores.items():
            if isinstance(value, dict):
                print(f"{name} : mean: {value['mean']:0.2f}, std: {value['std']:0.2f}, {value['CI']:0.2f} confidence interval: [{value['lower']:0.2f}, {value['upper']:0.2f}]")
            else:
                print(f"{name} : {value}")