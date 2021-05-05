#!/usr/bin/env python
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

parser = argparse.ArgumentParser(
    prog='dmatch',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')

subparsers = parser.add_subparsers(title='sub-command')
init_parser = subparsers.add_parser(
    'init',
    help='initialize alignment environment',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
prepare_parser = subparsers.add_parser(
    'prepare',
    help='prepare alignement dataset from 2 sources',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
align_parser = subparsers.add_parser(
    'align',
    help='align 2 data sources',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

def check_datasource(identifier):
    if identifier not in make_index.CONFIG:
        raise ValueError("Unkown data source identifier", identifier)
    if identifier not in make_index.CONNECTORS:
        raise ValueError("No connector defined for data source identifier", identifier)

def check_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), path
        )

# INIT SUBCOMMAND
def init_cmd(args):
    dirpath = os.path.abspath(args.dir)
    sourcepath = os.path.dirname(os.path.dirname(__file__))
    log.info(f'Initialize {dirpath}')
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    log.info(f'Copy connections template')
    configpath = os.path.join(sourcepath, 'connections.cfg.template')
    shutil.copyfile(configpath, os.path.join(dirpath, 'connections.cfg'))

    log.info(f'Copy connectors template')
    connectorpath = os.path.join(sourcepath, 'connectors.py')
    shutil.copyfile(connectorpath, os.path.join(dirpath, 'connectors.py'))

init_parser.add_argument('dir', nargs='?', default=os.getcwd())
init_parser.set_defaults(func=init_cmd)

# PREPARE SUBCOMMAND
def prepare_cmd(args):
    check_datasource(args.id1)
    check_datasource(args.id2)

    dst = os.path.relpath(args.dst)
    index1 = os.path.join(dst, 'index1')
    index2 = os.path.join(dst, 'index2')

    if args.filter1:
        filterpath1 = os.path.relpath(args.filter1)
        check_file(filterpath1)
    else:
        filterpath1 = None
    if args.filter2:
        filterpath2 = os.path.relpath(args.filter2)
        check_file(filterpath2)
    else:
        filterpath2 = None

    alignement = os.path.join(dst, 'alignment')
    make_index.make_index(index1, args.id1, args.m, args.n)
    make_index.make_index(index2, args.id2, args.m, args.n)
    preprocess.preprocess(index1, filterpath1)
    preprocess.preprocess(index2, filterpath2)
    make_correspondances.make_correspondances(index1, index2, alignement)
    process.process(alignement)
    
prepare_parser.add_argument('id1', help='data source1 identifier')
prepare_parser.add_argument('id2', help='data source2 identifier')
prepare_parser.add_argument('dst', help='path where alignment of the two indexes will be stored, will be erased if exists')
prepare_parser.add_argument(
    '-m',
    metavar='mode',
    type=str,
    choices=['random', 'percentile', 'interpolate', 'full'],
    default='interpolate',
    help='sampling mode {random, percentile, interpolate, full}'
)
prepare_parser.add_argument('-n', type=int, default=1000, help='sampling size')
prepare_parser.add_argument('--filter1', type=str, help='path to a filter file')
prepare_parser.add_argument('--filter2', type=str, help='path to a filter file')
prepare_parser.set_defaults(func=prepare_cmd)


# ALIGN SUBCOMMAND
def align_cmd(args):
    check_datasource(args.id1)
    check_datasource(args.id2)

    dst = os.path.relpath(args.dst)
    index1 = os.path.join(dst, 'index1')
    index2 = os.path.join(dst, 'index2')

    if args.filter1:
        filterpath1 = os.path.relpath(args.filter1)
        check_file(filterpath1)
    else:
        filterpath1 = None
    if args.filter2:
        filterpath2 = os.path.relpath(args.filter2)
        check_file(filterpath2)
    else:
        filterpath2 = None

    model = os.path.relpath(args.matcher)
    check_file(model)

    alignement = os.path.join(dst, 'alignment')
    make_index.make_index(index1, args.id1, args.m, args.n)
    make_index.make_index(index2, args.id2, args.m, args.n)
    preprocess.preprocess(index1, filterpath1)
    preprocess.preprocess(index2, filterpath2)
    make_correspondances.make_correspondances(index1, index2, alignement)
    process.process(alignement)
    match.match(alignement, model)
    
align_parser.add_argument('id1', help='data source1 identifier')
align_parser.add_argument('id2', help='data source2 identifier')
align_parser.add_argument('matcher', help='path to the decision model')
align_parser.add_argument('dst', help='path where alignment of the two indexes will be stored, will be erased if exists')
align_parser.add_argument(
    '-m',
    metavar='mode',
    type=str,
    choices=['random', 'percentile', 'interpolate', 'full'],
    default='interpolate',
    help='sampling mode {random, percentile, interpolate, full}'
)
align_parser.add_argument('-n', type=int, default=1000, help='sampling size')
align_parser.add_argument('--filter1', type=str, help='path to a filter file')
align_parser.add_argument('--filter2', type=str, help='path to a filter file')
align_parser.set_defaults(func=align_cmd)