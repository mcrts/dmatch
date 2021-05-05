#!/usr/bin/env python3.8
# -*- coding: utf-8 -*
import argparse
import os
import errno
import shutil

from ..__version__ import __version__
from ..logger import log

from .. import make_index 
from .. import preprocess
from .. import make_correspondances
from .. import process
from .. import match
from .. import resample

parser = argparse.ArgumentParser(
    prog='dmatch-tools',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')

subparsers = parser.add_subparsers(title='sub-command')
index_parser = subparsers.add_parser(
    'index',
    help='extract terminology and sample data from source into an index',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
preprocess_parser = subparsers.add_parser(
    'preprocess',
    help='preprocess the index by filtering terminology and computing KDEs',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
prepare_parser = subparsers.add_parser(
    'prepare',
    help='prepare all correspondances between two indexes',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
score_parser = subparsers.add_parser(
    'score',
    help='process all correspondances between two indexes',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
match_parser = subparsers.add_parser(
    'match',
    help='produce alignment from scored correspondances',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
resample_parser = subparsers.add_parser(
    'resample',
    help='resample source index data into a new index',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)


# INDEX SUBCOMMAND
def index_cmd(args):
    if args.identifier not in dmatch.make_index.CONFIG:
        raise ValueError("Unkown data source identifier", args.identifier)
    if args.identifier not in dmatch.make_index.CONNECTORS:
        raise ValueError("No connector defined for data source identifier", args.identifier)
    dst = os.path.relpath(args.dst)
    if os.path.exists(dst):
        log.info(f'Destination folder {dst} already exists and will be erased')
        shutil.rmtree(dst)
    make_index.make_index(dst, args.identifier, args.m, args.n)

index_parser.add_argument('identifier', help='data source identifier')
index_parser.add_argument('dst', help='path where index will be stored, will be erased if exists')
index_parser.add_argument(
    '-m',
    metavar='mode',
    type=str,
    choices=['random', 'percentile', 'interpolate', 'full'],
    default='interpolate',
    help='sampling mode {random, percentile, interpolate, full}'
)
index_parser.add_argument('-n', type=int, default=1000, help='sampling size')
index_parser.set_defaults(func=index_cmd)


# PREPROCESS SUBCOMMAND
def preprocess_cmd(args):
    src = os.path.relpath(args.src)
    if not os.path.exists(src):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), src
        )
    
    if args.filter:
        filterpath = os.path.relpath(args.filter)
        if not os.path.exists(filterpath):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), filterpath
            )
    else:
        filterpath = None
    preprocess.preprocess(src, filterpath)

preprocess_parser.add_argument('src', help='path to index')
preprocess_parser.add_argument('--filter', type=str, help='path to a filter file')
preprocess_parser.set_defaults(func=preprocess_cmd)


# PREPARE SUBCOMMAND
def prepare_cmd(args):
    index1 = os.path.relpath(args.index1)
    index2 = os.path.relpath(args.index2)
    if not os.path.exists(index1):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), index1
        )
    if not os.path.exists(index2):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), index2
        )
    dst = os.path.relpath(args.dst)
    if os.path.exists(dst):
        log.info(f'Destination folder {dst} already exists and will be erased')
        shutil.rmtree(dst)
    make_correspondances.make_correspondances(index1, index2, dst)

prepare_parser.add_argument('index1', help='path to index one')
prepare_parser.add_argument('index2', help='path to index two')
prepare_parser.add_argument('dst', help='path where alignment of the two indexes will be stored, will be erased if exists')
prepare_parser.set_defaults(func=prepare_cmd)


# SCORE SUBCOMMAND
def score_cmd(args):
    index = os.path.relpath(args.index)
    if not os.path.exists(index):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), index
        )
    process.process(index)

score_parser.add_argument('index', help='path to the alignment folder')
score_parser.set_defaults(func=score_cmd)

# MATCH SUBCOMMAND
def match_cmd(args):
    index = os.path.relpath(args.index)
    model = os.path.relpath(args.model)
    if not os.path.exists(index):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), index
        )
    if not os.path.exists(model):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), model
        )
    match.match(index, model)

match_parser.add_argument('index', help='path to the alignment folder')
match_parser.add_argument('model', help='path to the decision model')
match_parser.set_defaults(func=match_cmd)

# RESAMPLE SUBCOMMAND
def resample_cmd(args):
    src = os.path.relpath(args.src)
    dst = os.path.relpath(args.dst)
    size = args.n
    if src == dst:
        raise ValueError("Source and Destination must be different", src, dst)
    if not os.path.exists(src):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), src
        )
    if os.path.exists(dst):
        log.info(f'Destination folder {dst} already exists and will be erased')
        shutil.rmtree(dst)
    resample.resample(src, dst, size)

resample_parser.add_argument('src', help='path to source index to resample from')
resample_parser.add_argument('dst', help='path to destination where resample index will be stored, will be erased if exists')
resample_parser.add_argument('-n', type=int, default=1000, help='sampling size')
resample_parser.set_defaults(func=resample_cmd)