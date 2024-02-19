#! /bin/bash

CUDA_VISIBLE_DEVICES=0 python scripts/test.py --cfg configs/efn4_fpn_hm_adv.yaml \
                                              -i 447.png
