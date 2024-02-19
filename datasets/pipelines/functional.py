#-*- coding: utf-8 -*-
import numpy as np


def _get_pixels(per_pixel, rand_color, patch_size, dtype=np.float32):
    if per_pixel:
        return np.random.randint(0, 255, patch_size).astype(dtype=dtype)
    elif rand_color:
        return np.random.randint(0, 255, (1, 1, patch_size[2])).astype(dtype=dtype)
    else:
        return np.zeros((1, 1, patch_size[2]), dtype=dtype)
