#-*- coding: utf-8 -*-
#-*- coding: utf-8 -*-
from typing import Any, Dict, Optional

from register.register import Registry, build_from_cfg


LOSSES = Registry('Loss')


def build_losses(cfg,
                 loss_func: Registry,
                 build_func=build_from_cfg,
                 default_args: Optional[Dict] = None) -> Any:
    return build_func(cfg, loss_func, default_args)
