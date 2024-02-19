#-*- coding: utf-8 -*-
from abc import abstractmethod
import os
from random import shuffle
import sys

import torch
from PIL import Image
import numpy as np

from .builder import DATASETS, PIPELINES, build_pipeline
from .common import CommonDataset


@DATASETS.register_module()
class BinaryFaceForensic(CommonDataset):
    def __init__(self,
                 config,
                 split,
                 **kwargs):
        """
        @params:
        config: Dataset config
        split: train/val/test which directs to the split folders
        """
        self.split = split
        if kwargs is not None:
            for k,v in kwargs.items():
                if v is None:
                    raise ValueError(f'{k}:{v} retrieve a None value!')
                self.__setattr__(k, v)
        super().__init__(config, **kwargs)

        #Load data
        self.image_paths, self.labels, self.mask_paths = self._load_data(split)
        
        #Calling transform methods for inputs
        self.geo_transform = build_pipeline(config.TRANSFORM.geometry, PIPELINES)
        self.colorjitter_transform = build_pipeline(config.TRANSFORM.color, PIPELINES)

    def _load_data(self, split):
        from_file = self._cfg.FROM_FILE
        
        if not from_file:
            image_paths, labels, mask_paths = self._load_from_path(split)
        else:
            image_paths, labels, mask_paths = self._load_from_file(split)
        
        assert len(image_paths) != 0, "Image paths have not been loaded! Please check image directory!"
        assert len(labels) != 0, "Labels have not been loaded! Please check image folder path!"
        return image_paths, labels, mask_paths

    def _load_img(self, img_path):
        img = Image.open(img_path)
        img = np.array(img)
        return img

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = np.expand_dims(self.labels[idx], axis=-1)
        img = self._load_img(img_path)

        #Applying geo transform to inputs
        geo_transfomed = self.geo_transform(img)
        img_trans = geo_transfomed['image']
        
        #Applying color transform to inputs
        color_transfomed = self.colorjitter_transform(img_trans)
        img_trans = color_transfomed['image']

        #Normalise + Convert numpy array to tensor
        img_trans = img_trans/255
        img_trans = self.final_transforms(img_trans)
        return img_trans, label


if __name__=="__main__":
    from datasets import *
    from pipelines.geo_transform import GeometryTransform
    from pipelines.color_transform import ColorJitterTransform
    from torch.utils.data import DataLoader
    from configs.get_config import load_config
    
    PIPELINES.register_module(module=GeometryTransform)
    PIPELINES.register_module(module=ColorJitterTransform)

    config = load_config("configs/base.yaml")
    bin_ff = DATASETS.build(cfg=config.DATASET, default_args=dict(split='val', config=config.DATASET))
    bin_ff_loader = DataLoader(bin_ff,
                               batch_size=2,
                               shuffle=True)
    for b, (X, y) in enumerate(bin_ff_loader):
        print(f'X.shape - {X.shape}, y shape - {y.shape}')
        break
