#!/usr/bin/env python
# -*- coding: utf-8 -*
import os
import csv
import itertools as it

import pandas as pd

from .utils import CSV_READ_FORMAT, CSV_WRITE_FORMAT
from .logger import log

def compute_correspondances(indexa, indexb):
    terminologya_path = os.path.join(indexa, 'terminology.csv')
    entitya = pd.read_csv(terminologya_path, **CSV_READ_FORMAT).entityid.apply(lambda e: str(e) + ':' + indexa)
    terminologyb_path = os.path.join(indexb, 'terminology.csv')
    entityb = pd.read_csv(terminologyb_path, **CSV_READ_FORMAT).entityid.apply(lambda e: str(e) + ':' + indexb)
    correspondances = it.product(entitya, entityb)
    df = pd.DataFrame(correspondances, columns=['entityA', 'entityB'])
    return df


def make_correspondances(indexa, indexb, destination):
    log.info("Create folder")
    os.makedirs(destination, exist_ok=True)
    log.info("Compute correspondances")
    df = compute_correspondances(indexa, indexb)
    df.to_csv(os.path.join(destination, 'correspondances.csv'), **CSV_WRITE_FORMAT)
