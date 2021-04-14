#!/usr/bin/env python
# -*- coding: utf-8 -*
import os
import importlib.util
from configparser import ConfigParser, ExtendedInterpolation

from .logger import log
from .connectors import CONNECTORS

# Kind of a hack to fetch connections config file outside of package
# Should probably set config file as a parameter
# Or design a Dmatch Object that implement all command
def get_config():
    CONFIGPATH = os.path.join(os.getcwd(), 'connections.cfg')
    CONFIG = ConfigParser(interpolation=ExtendedInterpolation())
    if os.path.exists(CONFIGPATH):
        CONFIG.read(CONFIGPATH)
    else:
        log.debug('No connections file found')
        CONFIG.read({})
    return CONFIG

def get_connectors():
    connector_path = os.path.join(os.getcwd(), 'connectors.py')
    if os.path.exists(connector_path):
        spec = importlib.util.spec_from_file_location("local_connectors", connector_path)
        connectors = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(connectors)
        all_connectors = {**CONNECTORS, **connectors.CONNECTORS}
    else:
        log.debug('No connectors file found')
        all_connectors = CONNECTORS
    return all_connectors
