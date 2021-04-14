#!/usr/bin/env python
# -*- coding: utf-8 -*
import os
import csv

import pandas as pd
import numpy as np


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