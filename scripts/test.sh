#! /bin/bash

CUDA_VISIBLE_DEVICES=0 python scripts/test.py --cfg configs/resnet_fpn_hm.yaml \
                                    -i /data/deepfake_cluster/datasets_df/FaceForensics++/c0/test/frames/Deepfakes/000_003/012.png
