#!/usr/bin/env python
# -*- coding: utf-8 -*
import os
import csv

import pandas as pd
import numpy as np
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from sklearn.preprocessing import scale
from scipy.stats import ks_2samp


from .utils import CSV_READ_FORMAT, CSV_WRITE_FORMAT
from .utils import Accessor, Stats
from .logger import log


def compute_aggregates(row):
    metadataA = Accessor.get_entity_aggregate(row.entityA)
    metadataB = Accessor.get_entity_aggregate(row.entityB)
    mean = abs(metadataA['mean'] - metadataB['mean'])
    std = abs(metadataA['std'] - metadataB['std'])
    var = abs(metadataA['var'] - metadataB['var'])
    frequency = abs(metadataA['frequency'] - metadataB['frequency'])
    result = pd.Series({'mean': mean, 'std': std, 'var': var, 'frequency': frequency})
    return result

def compute_hellinger_distance(row):
    hd = Accessor.hellinger_distance_2entity(row.entityA, row.entityB)
    result = pd.Series({'hellinger_distance': hd})
    return result

def compute_ks_test(row):
    ks, pvalue = Accessor.ks_test_2entity(row.entityA, row.entityB)
    result = pd.Series({'ks_test': ks, 'pvalue': pvalue})
    return result

def compute_scaled_hellinger_distance(row):
    dataA = Accessor.get_entity_data(row.entityA).reshape(-1, 1)
    scaled_dataA = scale(dataA).reshape(1, -1)
    dataB = Accessor.get_entity_data(row.entityB).reshape(-1, 1)
    scaled_dataB = scale(dataB).reshape(1, -1)
    hd = Stats.hellinger_distance_2samp(scaled_dataA, scaled_dataB)
    result = pd.Series({'hellinger_distance': hd})
    return result

def compute_scaled_ks_test(row):
    dataA = Accessor.get_entity_data(row.entityA).reshape(-1, 1)
    scaled_dataA = scale(dataA).flatten()
    dataB = Accessor.get_entity_data(row.entityB).reshape(-1, 1)
    scaled_dataB = scale(dataB).flatten()
    ks, pvalue = ks_2samp(scaled_dataA, scaled_dataB)
    result = pd.Series({'ks_test': ks, 'pvalue': pvalue})
    return result

def process(folder):
    df = pd.read_csv(os.path.join(folder, 'correspondances.csv'), **CSV_READ_FORMAT)
    ddf = dd.from_pandas(df, npartitions=16)

    log.info('Computing aggregates')
    with ProgressBar():
        res = ddf.apply(
            compute_aggregates,
            meta={'mean': float, 'std': float, 'var': float, 'frequency': float},
            result_type='expand',
            axis=1
        ).compute(scheduler='multiprocessing') 
    df['mean'] = res['mean']
    df['std'] = res['std']
    df['var'] = res['var']
    df['frequency'] = res['frequency']

    log.info('Computing Hellinger distance')
    with ProgressBar():
        res = ddf.apply(compute_hellinger_distance, meta={'hellinger_distance': float}, result_type='expand', axis=1).compute(scheduler='multiprocessing') 
    df['hellinger_distance'] = res['hellinger_distance']

    log.info('Computing Kolmogorov-Smirnov test')
    with ProgressBar():
        res = ddf.apply(compute_ks_test, meta={'ks_test': float, 'pvalue': float}, result_type='expand', axis=1).compute(scheduler='multiprocessing') 
    df['ks_test'] = res['ks_test']
    df['ks_pvalue'] = res['pvalue']

    log.info('Computing scaled Hellinger distance')
    with ProgressBar():
        res = ddf.apply(compute_scaled_hellinger_distance, meta={'hellinger_distance': float}, result_type='expand', axis=1).compute(scheduler='multiprocessing') 
    df['scaled_hellinger_distance'] = res['hellinger_distance']

    log.info('Computing scaled Kolmogorov-Smirnov test')
    with ProgressBar():
        res = ddf.apply(compute_scaled_ks_test, meta={'ks_test': float, 'pvalue': float}, result_type='expand', axis=1).compute(scheduler='multiprocessing') 
    df['scaled_ks_test'] = res['ks_test']
    df['scaled_ks_pvalue'] = res['pvalue']

    log.info('Saving results')
    df.to_csv(os.path.join(folder, 'scores.csv'), **CSV_WRITE_FORMAT)
