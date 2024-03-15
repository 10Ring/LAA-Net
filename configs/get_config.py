#-*- coding: utf-8 -*-
import os

from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper    
from box import Box as edict


def load_config(cfg):
    with open(cfg) as f:
        config = load(f, Loader=Loader)

    return edict(config)
