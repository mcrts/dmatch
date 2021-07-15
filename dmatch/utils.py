#!/usr/bin/env python
# -*- coding: utf-8 -*
import os
import csv

import cloudpickle
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import gaussian_kde, ks_2samp
from sklearn.feature_selection import SelectorMixin
from sklearn.base import TransformerMixin

from .logger import log

CSV_WRITE_FORMAT = {
    'index': False,
    'quoting': csv.QUOTE_ALL,
}
CSV_READ_FORMAT = {
    'keep_default_na': False,
}

class Sample:
    @staticmethod
    def full_sample(data, n):
        return data

    @staticmethod
    def random_sample(data, n, random_state=1):
        rng = np.random.default_rng(random_state)
        sample = rng.choice(data, n)
        return sample

    @staticmethod
    def percentile_sample(data, n, lower=0, upper=100):
        quantiles = np.linspace(lower, upper, n, endpoint=True)
        sample = np.percentile(data, quantiles, interpolation='lower')
        return sample

    @staticmethod
    def percentile_interpolation_sample(data, n, lower=0, upper=100):
        quantiles = np.linspace(lower, upper, n, endpoint=True)
        sample = np.percentile(data, quantiles, interpolation='linear')
        return sample
    
Sample.MODES = {
    'random': Sample.random_sample,
    'percentile': Sample.percentile_sample,
    'interpolate': Sample.percentile_interpolation_sample,
    'full': Sample.full_sample,
}


class Stats:
    @staticmethod
    def compute_integral_boundaries(f, retsize):
        u = f.resample(retsize)
        a = u.mean() - 8 * u.std()
        b = u.mean() + 8 * u.std()
        return a, b

    @staticmethod
    def discrete_hellinger_integral(p, q, a, b, retsize):
        x, step = np.linspace(a, b, retsize, endpoint=True, retstep=True)
        i = np.dot(np.sqrt(p(x)), np.sqrt(q(x))) * step
        if i > 1:
            return 0
        else:
            return i

    @classmethod
    def discrete_hellinger_distance(cls, p, q, retsize=100):
        a1, b1 = cls.compute_integral_boundaries(p, retsize)
        a2, b2 = cls.compute_integral_boundaries(q, retsize)
        a1, b1, a2, b2 = sorted([a1, b1, a2, b2])
        i1 = cls.discrete_hellinger_integral(p, q, a1, b1, retsize)
        i2 = cls.discrete_hellinger_integral(p, q, b1, a2, retsize)
        i3 = cls.discrete_hellinger_integral(p, q, a2, b2, retsize)
        i = i1 + i2 + i3
        if i > 1: # To prevent computing a negative root because of an approximation error during integration
            return 0
        else:
            return np.sqrt(1 - i)
    
    @staticmethod
    def hellinger_integral(p, q, a=-np.inf, b=np.inf):
        value, error = quad(
            lambda x: np.sqrt(p(x)*q(x)),
            a,
            b
        )
        return value, error

    @classmethod
    def hellinger_distance(cls, p, q, a=-np.inf, b=np.inf, split_integral=True, retsize=100):
        if split_integral:
            a1, b1 = cls.compute_integral_boundaries(p, retsize)
            a2, b2 = cls.compute_integral_boundaries(q, retsize)
            a1, b1, a2, b2 = sorted([a1, b1, a2, b2])
            i1, _ = cls.hellinger_integral(p, q, a1, b1)
            i2, _ = cls.hellinger_integral(p, q, b1, a2)
            i3, _ = cls.hellinger_integral(p, q, a2, b2)
            value = i1 + i2 + i3
        else:
            value, error = cls.hellinger_integral(p.pdf, q.pdf, a, b)

        if 1 <= value < 1.1: # To prevent computing a negative root because of an approximation error during integration
            return 1
        elif 1.1 <= value: # If value > 1.1 the approximation failed too much and should not be rejected
            return 0
        else:
            return np.sqrt(1 - value)
    
    @classmethod
    def hellinger_distance_1samp(cls, sample, pdf, **params):
        kde = gaussian_kde(sample, bw_method='silverman')
        return cls.hellinger_distance(kde, pdf, split_integral=False)

    @classmethod
    def hellinger_distance_2samp(cls, samp1, samp2):
        kde1 = gaussian_kde(samp1, bw_method='silverman')
        kde2 = gaussian_kde(samp2, bw_method='silverman')
        return cls.hellinger_distance(kde1, kde2)

class Accessor:
    @staticmethod
    def get_entity_kde(entity):
        entityid, indexid = entity.split(':')
        kdepath = os.path.join(indexid, 'kde', entityid + '.kde')
        with open(kdepath, 'rb') as f:
            kde = cloudpickle.load(f)
        return kde

    @staticmethod
    def get_entity_data(entity):
        entityid, indexid = entity.split(':')
        datapath = os.path.join(indexid, 'entity', entityid + '.npy')
        data = np.load(datapath)
        return data

    @staticmethod
    def get_entity_metadata(entity):
        entityid, indexid = entity.split(':')
        path = os.path.join(indexid, 'terminology.csv')
        terminology = pd.read_csv(path, **CSV_READ_FORMAT, dtype=str)
        data = terminology[terminology.entityid == entityid].squeeze()
        return data

    @staticmethod
    def get_entity_aggregate(entity):
        columntypes = {
            "entityid": str,
            "size": int,
            "mean": float,
            "std": float,
            "var": float,
            "frequency": float
        }
        entityid, indexid = entity.split(':')
        path = os.path.join(indexid, 'aggregate.csv')
        terminology = pd.read_csv(path, **CSV_READ_FORMAT, dtype=columntypes)
        data = terminology[terminology.entityid == entityid].squeeze()
        return data

    @classmethod
    def hellinger_distance_2entity(cls, entity1, entity2, strategy='split_integral'):
        kde1 = cls.get_entity_kde(entity1)
        kde2 = cls.get_entity_kde(entity2)
        strategies = ('full', 'split_integral', 'discrete')
        if strategy not in strategies:
            strategy = 'split_integral'
            log.info(f"Hellinger distance strategy {strategy} must be in {strategies}, switching to 'split_integral'")

        if strategy == 'full':
            hd = Stats.hellinger_distance(kde1, kde2, split_integral=False)
        elif strategy == 'split_integral':
            hd = Stats.hellinger_distance(kde1, kde2, split_integral=True)
        elif strategy == 'discrete':
            hd = Stats.discrete_hellinger_distance(kde1, kde2)
        return hd

    @classmethod
    def ks_test_2entity(cls, entity1, entity2):
        data1 = cls.get_entity_data(entity1).flatten()
        data2 = cls.get_entity_data(entity2).flatten()
        return ks_2samp(data1, data2)

    @classmethod
    def kde_from_entity(cls, entity):
        entityid, indexid = entity.split(':')
        kdepath = os.path.join(indexid, 'kde', entityid + '.kde')
        os.makedirs(os.path.dirname(kdepath), exist_ok=True)

        data = cls.get_entity_data(entity)
        kde = gaussian_kde(data, bw_method='silverman')
        with open(kdepath, 'wb') as f:
            cloudpickle.dump(kde, f)  
        return kde

class CachedAccessor:
    KDECACHE = dict()
    @classmethod
    def get_entity_kde(cls, entity):
        if entity in cls.KDECACHE:
            kde = cls.KDECACHE.get(entity)
        else:
            kde = Accessor.get_entity_kde(entity)
            cls.KDECACHE[entity] = kde
        return kde

    DATACACHE = dict()
    @classmethod
    def get_entity_data(cls, entity):
        if entity in cls.DATACACHE:
            data = cls.DATACACHE.get(entity)
        else:
            data = Accessor.get_entity_data(entity)
            cls.DATACACHE[entity] = data
        return data

    METDATACACHE = dict()
    @classmethod
    def get_entity_metadata(cls, entity):
        if entity in cls.METDATACACHE:
            data = cls.METDATACACHE.get(entity)
        else:
            data = Accessor.get_entity_metadata(entity)
            cls.METDATACACHE[entity] = data
        return data


class Score:
    @staticmethod
    def tm_rank(groupby, ret=5):
        return groupby.sort_values('y_proba', ascending=False).head(ret).Y.any()

    @staticmethod
    def tm_score(df, ret=5):
        res = df.groupby('entityA').apply(Score.tm_rank, ret=ret)
        n = len(df.index.get_level_values('entityA').unique())
        return res.value_counts().get(True, 0) / n
    
    @staticmethod
    def tm_score_relaxed(df, ret=5):
        res = df.groupby('entityA').apply(Score.tm_rank, ret=ret)
        n = len(df[df.Y == True].index.get_level_values('entityA').unique())
        return res.value_counts().get(True, 0) / n
    
    @staticmethod
    def compute_tm_score(model, df, ret=5):
        df = df.copy()
        df['y_proba'] = model.predict_proba(df.X)[:,1]
        res = df.groupby('entityA').apply(Score.tm_rank, ret=ret)
        n = len(df[df.Y == True].index.get_level_values('entityA').unique())
        return res.value_counts().get(True, 0) / n


class NamedFeatureSelector(SelectorMixin, TransformerMixin):
    def __init__(self, columns, selected_columns):
        self.columns = columns
        self.selected_columns = set(selected_columns)

    def _get_support_mask(self):
        mask = np.array(list(map(lambda x: x in self.selected_columns, self.columns)))
        return mask
    
    def fit(self, X, y=None):
        return self


from sklearn.utils import resample
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.stats import t

class Bootstrap:
    @staticmethod
    def sample(df, rate=1):
        n = int(len(df.index) * rate)
        return resample(df, n_samples=n)
    
    @staticmethod
    def evaluate(model, data):
        X = data.drop('Y', axis=1)
        Y = data['Y']
        Y_pred = model.predict(X)
        stats = (f1_score(Y, Y_pred, zero_division=0), precision_score(Y, Y_pred, zero_division=0), recall_score(Y, Y_pred, zero_division=0))
        return stats
    
    @staticmethod
    def score(model, df, rep=1000, rate=1, verbose=False):
        statistics = []
        for i in range(rep):
            if verbose and (i % 50 == 0):   
                log.info(f"Bootstrap iteration {i} over {rep}")

            test = Bootstrap.sample(df, rate)
            stat = Bootstrap.evaluate(model, test)
            statistics.append(stat)
        
        statistics = np.array(statistics)
        results = dict()
        for name, stats in zip(['f1', 'precision', 'recall'], statistics.T):
            mu = stats.mean()
            std = stats.std()
            alpha = 0.05
            if std > 0:
                st = (mu - stats) / std
                q1 = mu - np.quantile(st, 1-0.5*alpha)*std
                q2 = mu - np.quantile(st, 0.5*alpha)*std
            else:
                q1 = q2 = mu
            results[name] = {
                'mean': mu,
                'std': std,
                'CI': 1-alpha,
                'lower': q1,
                'upper': q2,
            }
        return results