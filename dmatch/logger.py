#!/usr/bin/env python
# -*- coding: utf-8 -*
import sys
import logging
log = logging.getLogger("dmatch")
log.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
log.addHandler(ch)