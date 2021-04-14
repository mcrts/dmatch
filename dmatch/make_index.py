#!/usr/bin/env python
# -*- coding: utf-8 -*
import os
import csv

import numpy as np
import pandas
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

from .utils import CSV_WRITE_FORMAT
from .utils import Sample
from .logger import log
from .config import get_config, get_connectors

CONFIG = get_config()
CONNECTORS = get_connectors()


def sample(row, connid, uri, sample_mode, size, entitypath):
    connector = CONNECTORS[connid](uri)
    sampler = Sample.MODES.get(sample_mode, Sample.percentile_interpolation_sample)
    array = connector.get_entities(row.entityid)
    array = sampler(array, size)
    np.save(os.path.join(entitypath, f'{row.entityid}.npy'), array)
    return 1

def make_index(indexpath, connid, mode='percentile_interpolation_sample', size=1000):
    entitypath = os.path.join(indexpath, 'entity')
    log.info("Create folder scafolding")
    os.makedirs(indexpath, exist_ok=True)
    os.makedirs(entitypath, exist_ok=True)

    uri = CONFIG.get(connid, 'uri')
    connector = CONNECTORS[connid](uri)

    log.info(f"Fetch terminology for {connid} data source")
    terminology = connector.get_terminology()
    terminology.to_csv(os.path.join(indexpath, 'terminology.csv'), **CSV_WRITE_FORMAT)
    
    log.info("Fetch data statistics")
    aggregate = connector.get_terminology_aggregate()
    aggregate.to_csv(os.path.join(indexpath, 'aggregate.csv'), **CSV_WRITE_FORMAT)
    
    log.info("Sample data")
    ddf = dd.from_pandas(terminology, npartitions=32)
    with ProgressBar():
        ddf.apply(sample, args=(connid, uri, mode, size, entitypath), meta=('x', 'int'), axis=1).compute(scheduler='multiprocessing')
