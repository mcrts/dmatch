#!/usr/bin/env python
# -*- coding: utf-8 -*
import os
import csv

import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

from .utils import CSV_READ_FORMAT, CSV_WRITE_FORMAT
from .utils import Stats, Accessor
from .logger import log

def compute_kde(row):
    Accessor.kde_from_entity(row)
    return 1

def filter_index(indexfolder, filterpath):
    entityids = set(pd.read_csv(filterpath, **CSV_READ_FORMAT).entityid)
    terminologypath  = os.path.join(indexfolder, 'terminology.csv')
    terminology = pd.read_csv(terminologypath, **CSV_READ_FORMAT)
    terminology = terminology[terminology['entityid'].isin(entityids)]

    aggregatepath  = os.path.join(indexfolder, 'aggregate.csv')
    aggregate = pd.read_csv(aggregatepath, **CSV_READ_FORMAT)
    aggregate = aggregate[aggregate['entityid'].isin(entityids)]
    return terminology, aggregate

def preprocess(indexfolder, filterpath=None):
    log.info("Filter terminology")
    terminologypath  = os.path.join(indexfolder, 'terminology.csv')
    aggregatepath  = os.path.join(indexfolder, 'aggregate.csv')
    if filterpath:
        terminology, aggregate = filter_index(indexfolder, filterpath)
        terminology.to_csv(terminologypath, **CSV_WRITE_FORMAT)
        aggregate.to_csv(aggregatepath, **CSV_WRITE_FORMAT)
    else:
        terminology = pd.read_csv(terminologypath, **CSV_READ_FORMAT)
    
    log.info("Compute KDEs")
    entityids = terminology['entityid'].apply(lambda e: str(e) + ':' + indexfolder)
    ddf = dd.from_pandas(entityids, npartitions=32)
    with ProgressBar():
        ddf.apply(Accessor.kde_from_entity, meta=('x', 'int')).compute(scheduler='multiprocessing')