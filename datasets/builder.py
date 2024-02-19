#-*- coding: utf-8 -*-
import os
import sys
from typing import Any, Optional, Dict
if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

from register.register import Registry, build_from_cfg


PIPELINES = Registry('Pipeline', build_func=build_from_cfg)
DATASETS = Registry('Dataset', build_func=build_from_cfg)


def build_pipeline(cfg, 
                   pipeline: Registry , 
                   build_func=build_from_cfg, 
                   default_args: Optional[Dict] = None) -> Any:
    return build_func(cfg, pipeline, default_args)


def build_dataset(cfg,
                  dataset: Registry,
                  build_func=build_from_cfg, 
                  default_args: Optional[Dict] = None) -> Any:
    return build_func(cfg, dataset, default_args)
