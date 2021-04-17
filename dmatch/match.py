#!/usr/bin/env python
# -*- coding: utf-8 -*
import os
import csv

import pandas as pd
import numpy as np
from joblib import load

from .utils import CSV_READ_FORMAT, CSV_WRITE_FORMAT
from .utils import Stats, Accessor
from .logger import log

CSV_WRITE_FORMAT = dict(CSV_WRITE_FORMAT)
CSV_WRITE_FORMAT['index'] = True

def make_report(row):
    entityA, entityB = row.name
    metadataA = Accessor.get_entity_metadata(entityA)
    metadataB = Accessor.get_entity_metadata(entityB)
    result = pd.concat([pd.Series({'Probability': row.ProbabilityTrue}), metadataA.add_suffix("A"), metadataB.add_suffix("B")], axis=0)
    return result

def match(index, modelpath):
    log.info(f'Load model {modelpath}')
    model = load(modelpath)
    df = pd.read_csv(os.path.join(index, 'scores.csv'), **CSV_READ_FORMAT, index_col=[0, 1])
    log.info('Evaluate alignment')

    prediction = model.predict(df)
    probability_rate = model.predict_proba(df)
    probability_true = probability_rate[:,1]
    probability_false = probability_rate[:,0]

    df['Prediction'] = prediction
    df['ProbabilityTrue'] = probability_true
    df['ProbabilityFalse'] = probability_false

    df[['Prediction', 'ProbabilityTrue', 'ProbabilityFalse']].to_csv(os.path.join(index, 'predictions.csv'), **CSV_WRITE_FORMAT)
    alignement = df[df.Prediction == True]
    reportpath = os.path.join(index, 'alignement_report.csv')
    log.info(f'Build report {reportpath}')
    report = alignement.apply(make_report, axis=1, result_type='expand')
    report.to_csv(reportpath, **CSV_WRITE_FORMAT)