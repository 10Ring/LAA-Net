#! /bin/bash

CUDA_VISIBLE_DEVICES=0 python scripts/train.py --cfg configs/efn4_fpn_hm_adv.yaml
