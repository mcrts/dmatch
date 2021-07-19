#!/usr/bin/env python
# -*- coding: utf-8 -*
import os
import csv

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


from .utils import CSV_READ_FORMAT, CSV_WRITE_FORMAT
from .utils import Stats, Accessor


def resample(source_index, target_index, size=1000):
    os.makedirs(target_index, exist_ok=True)
    
    terminology = pd.read_csv(os.path.join(source_index, 'terminology.csv'), **CSV_READ_FORMAT)
    aggregate = pd.read_csv(os.path.join(source_index, 'aggregate.csv'), **CSV_READ_FORMAT)

    terminology.to_csv(os.path.join(target_index, 'terminology.csv'), **CSV_WRITE_FORMAT)
    
    entitypath = os.path.join(target_index, 'entity')
    os.makedirs(entitypath, exist_ok=True)
    for i, (_, row) in enumerate(terminology.iterrows()):
        print(f'Resampling {i}-th entity')
        entity = '{}:{}'.format(row.entityid, source_index)
        kde = Accessor.kde_from_entity(entity)
        array = kde.resample(size)
        np.save(os.path.join(entitypath, f'{row.entityid}.npy'), array)
        
        mask = aggregate.entityid == row.entityid
        aggregate.loc[mask, 'mean'] = array.mean()
        aggregate.loc[mask, 'std'] = array.std()
        aggregate.loc[mask, 'var'] = array.var()
    
    aggregate.to_csv(os.path.join(target_index, 'aggregate.csv'), **CSV_WRITE_FORMAT)


def make_train_test(source_index, train_dst, test_dst, train_sample_ratio):
    terminology = pd.read_csv(os.path.join(source_index, 'terminology.csv'), **CSV_READ_FORMAT)

    os.makedirs(train_dst, exist_ok=True)
    terminology.to_csv(os.path.join(train_dst, 'terminology.csv'), **CSV_WRITE_FORMAT)
    train_aggregate = pd.read_csv(os.path.join(source_index, 'aggregate.csv'), **CSV_READ_FORMAT)
    train_entitypath = os.path.join(train_dst, 'entity')
    os.makedirs(train_entitypath, exist_ok=True)

    os.makedirs(test_dst, exist_ok=True)
    terminology.to_csv(os.path.join(test_dst, 'terminology.csv'), **CSV_WRITE_FORMAT)
    test_aggregate = pd.read_csv(os.path.join(source_index, 'aggregate.csv'), **CSV_READ_FORMAT)
    test_entitypath = os.path.join(test_dst, 'entity')
    os.makedirs(test_entitypath, exist_ok=True)

    for i, (_, row) in enumerate(terminology.iterrows()):
        print(f'Train Test Split {i}-th entity')
        entity = '{}:{}'.format(row.entityid, source_index)
        data = Accessor.get_entity_data(entity)
        train_sample, test_sample = train_test_split(data, train_size=train_sample_ratio)

        np.save(os.path.join(train_entitypath, f'{row.entityid}.npy'), train_sample)
        np.save(os.path.join(test_entitypath, f'{row.entityid}.npy'), test_sample)
        
        mask = train_aggregate.entityid == row.entityid
        train_aggregate.loc[mask, 'mean'] = train_sample.mean()
        train_aggregate.loc[mask, 'std'] = train_sample.std()
        train_aggregate.loc[mask, 'var'] = train_sample.var()

        mask = test_aggregate.entityid == row.entityid
        test_aggregate.loc[mask, 'mean'] = test_sample.mean()
        test_aggregate.loc[mask, 'std'] = test_sample.std()
        test_aggregate.loc[mask, 'var'] = test_sample.var()

    train_aggregate.to_csv(os.path.join(train_dst, 'aggregate.csv'), **CSV_WRITE_FORMAT)
    test_aggregate.to_csv(os.path.join(test_dst, 'aggregate.csv'), **CSV_WRITE_FORMAT)