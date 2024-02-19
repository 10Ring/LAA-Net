#-*- coding: utf-8 -*-
import os
import sys
import random

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2

from .builder import DATASETS, PIPELINES, build_pipeline
from .master import MasterDataset
from .sbi.utils import *
from package_utils.transform import get_affine_transform, get_center_scale, draw_landmarks
from package_utils.utils import vis_heatmap
from package_utils.image_utils import load_image, crop_by_margin


@DATASETS.register_module()
class SBIFaceForensic(MasterDataset):
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
        super(SBIFaceForensic, self).__init__(config, **kwargs)
        if kwargs is not None:
            for k,v in kwargs.items():
                if v is None:
                    raise ValueError(f'{k}:{v} retrieve a None value!')
                self.__setattr__(k, v)
        self.rot = 0
        self.pixel_std = 200
        self.target_w = self._cfg.IMAGE_SIZE[1]
        self.target_h = self._cfg.IMAGE_SIZE[0]
        self.aspect_ratio = self.target_w * 1.0 / self.target_h
        self.sigma = self._cfg.SIGMA
        self.heatmap_type = self._cfg.HEATMAP_TYPE
        self.debug = self._cfg.DEBUG
        # self.train = self._cfg.TRAIN
        self.dynamic_fxray = self._cfg.DYNAMIC_FXRAY
        
        #Load data
        self.image_paths_r, self.labels_r, self.mask_paths_r, self.ot_props_r = self._load_data(split)

        #Calling transform methods for inputs
        self.geo_transform = build_pipeline(config.TRANSFORM.geometry, PIPELINES, 
            default_args={"additional_targets": {"image_f": "image", "mask_f": "mask"}})
        # self.colorjitter_transform = build_pipeline(config.TRANSFORM.color, PIPELINES)
        
        self.transforms = get_transforms()
    
    def __len__(self):
        return len(self.labels_r)
    
    def _load_img(self, img_path):
        return load_image(img_path)

    def _reload_data(self):
        self.image_paths, self.labels, self.mask_paths, self.ot_props = self._load_data(self.split)

    def _load_data(self, split, anno_file=None):
        from_file = self._cfg.DATA[self.split.upper()].FROM_FILE
        
        if not from_file:
            image_paths, labels, mask_paths, ot_props = self._load_from_path(split)
        else:
            image_paths, labels, mask_paths, ot_props = self._load_from_file(split, anno_file=anno_file)
        
        assert len(image_paths) != 0, "Image paths have not been loaded! Please check image directory!"
        assert len(labels) != 0, "Labels have not been loaded! Please check annotation file!"
        if not self.dynamic_fxray:
            assert len(mask_paths) != 0, "Mask paths have not been loaded! Please check mask directory!"
            
        if self.sampler_active:
            print('Running sampler...')
            params = dict(mask_paths=mask_paths, ot_props=ot_props)
            data_sampler = self._sampler(image_paths, labels, **params)
            image_paths, labels = data_sampler['image_paths'], data_sampler['labels']
            if len(mask_paths):
                mask_paths = data_sampler['mask_paths']
            if len(ot_props):
                ot_props = data_sampler['ot_props']
            print(f'n samples after running sampling --- {len(image_paths)}')
        return image_paths, labels, mask_paths, ot_props

    def __getitem__(self, idx):
        flag = True
        while flag:
            try:
                #Selecting data from data list
                img_path = self.image_paths_r[idx]
                label = self.labels_r[idx]
                vid_id = img_path.split('/')[-2]
                img = self._load_img(img_path)
                if self.split == 'test':
                    img = crop_by_margin(img, margin=[5, 5])
                
                img_f = None
                mask = None
                mask_f = None
                
                # if not self.dynamic_fxray or self.split == 'val':
                if not self.dynamic_fxray:
                    if bool(self.mask_paths_r):
                        mask_path = self.mask_paths_r[idx]
                        mask = self._load_img(mask_path)
                    else:
                        mask = np.zeros((img.shape[0], img.shape[1], 3))
                else:
                    if self.train:
                        if len(self.ot_props_r[idx]['aligned_lms']):
                            f_lms = self.ot_props_r[idx]['aligned_lms']
                        elif len(self.ot_props_r[idx]['orig_lms']):
                            f_lms = self.ot_props_r[idx]['orig_lms']
                        else:
                            f_lms = []
                        f_lms = np.array(f_lms)
                        if not f_lms.any():
                            raise ValueError('Can not find fake copy image of empty landmarks!')

                        if len(f_lms) > 68:
                            f_lms = reorder_landmark(f_lms)
                            
                        if self.debug:
                            img_lms_draw = draw_landmarks(img, f_lms)
                            Image.fromarray(img_lms_draw).save(f'samples/debugs/orig_{idx}_lms.jpg')
                        
                        if self.split == 'train':
                            if np.random.rand() < 0.5:
                                img, ___, f_lms, __ = sbi_hflip(img, None, f_lms, None)
                    
                        margin = np.random.randint(5, 25)
                        img_f, mask_f, img, mask = gen_target(img, f_lms, margin=[margin, margin])
                target = None
                target_f = None
                
                if mask is not None:
                    assert (mask.shape[:2] == img.shape[:2]), "Color Image and Mask must have the same shape!"
                
                # Applying affine transform
                c, s = get_center_scale(img.shape[:2], self.aspect_ratio, pixel_std=self.pixel_std)
                
                # Applying geo transform to images and masks
                if self.split == 'train':
                    geo_transfomed = self.geo_transform(img, mask=mask, image_f=img_f, mask_f=mask_f)
                    img = geo_transfomed['image']
                    mask = geo_transfomed['mask']
                    img_f = geo_transfomed['image_f']
                    mask_f = geo_transfomed['mask_f']
                
                trans = get_affine_transform(c, s, self.rot, self._cfg.IMAGE_SIZE, pixel_std=self.pixel_std)
                trans_heatmap = get_affine_transform(c, s, self.rot, self._cfg.HEATMAP_SIZE, pixel_std=self.pixel_std)
                
                input = cv2.warpAffine(img,
                                       trans,
                                       (int(self._cfg.IMAGE_SIZE[0]), int(self._cfg.IMAGE_SIZE[1])),
                                       flags=cv2.INTER_LINEAR)
                
                if img_f is not None:
                    input_f = cv2.warpAffine(img_f,
                                             trans,
                                             (int(self._cfg.IMAGE_SIZE[0]), int(self._cfg.IMAGE_SIZE[1])),
                                             flags=cv2.INTER_LINEAR)    
                
                if mask is not None:
                    target = cv2.warpAffine(mask,
                                            trans_heatmap,
                                            (int(self._cfg.HEATMAP_SIZE[0]), int(self._cfg.HEATMAP_SIZE[1])),
                                            flags=cv2.INTER_LINEAR)
                
                if mask_f is not None:
                    target_f = cv2.warpAffine(mask_f,
                                              trans_heatmap,
                                              (int(self._cfg.HEATMAP_SIZE[0]), int(self._cfg.HEATMAP_SIZE[1])),
                                              flags=cv2.INTER_LINEAR)
                
                #Target encoding
                #0 for original, 1 for FXRay, 2 for NoFXRay. If 2, comment: mask = (1 - mask) * mask * 4
                heatmap, cstency_hm = self.select_encode_method(version=1)(target) if (target is not None and self.heatmap_type=='gaussian' and self.train) else (None, None)
                heatmap_f, cstency_hm_f = self.select_encode_method(version=1)(target_f) if (target_f is not None and self.heatmap_type=='gaussian' and self.train) else (None, None)
                
                # Applying transform for blending images
                if self.train:
                    if target_f is None:
                        transformed = self.transforms(image=input.astype('uint8'))
                        input=transformed['image']
                    else:
                        transformed = self.transforms(image=input.astype('uint8'),
                                                      image_f=input_f.astype('uint8'))
                        input=transformed['image']
                        input_f=transformed['image_f']
                
                if self.debug:
                    # Image.fromarray(input).save(f'samples/debugs/affine_{idx}.jpg')
                    Image.fromarray(input_f).save(f'samples/debugs/affine_f_{idx}.jpg')
                    # Image.fromarray(target).save(f'samples/debugs/mask_affine_{idx}.jpg')
                    Image.fromarray(target_f).save(f'samples/debugs/mask_affine_f_{idx}.jpg')
                    # Image.fromarray(mask).save(f'samples/debugs/mask_{idx}.jpg')
                    # Image.fromarray(mask_f).save(f'samples/debugs/mask_f_{idx}.jpg')
                    vis_heatmap(input, cstency_hm_f/255, f'samples/debugs/cstency_mask_f_{idx}.jpg')
                    # vis_heatmap(input, heatmap, f'samples/debugs/hm_{idx}.jpg')
                    vis_heatmap(input_f, heatmap_f, f'samples/debugs/hm_f_{idx}.jpg')
                
                if self.train:
                    # if self.split_image:
                    #     patch_img_trans = []
                    #     patch_heatmap_r = []
                    #     patch_img_trans_f = []
                    #     patch_heatmap_f = []
                    #     patch_target_r = []
                    #     patch_target_f = []
                        
                    #     for i, (k, l) in enumerate([[0,0], [1/2,0], [0,1/2], [1/2,1/2]]):
                    #         input_ = input[int(self.target_h*k): int(self.target_h*(k+1/2)), int(self.target_w*l): int(self.target_w*(l+1/2)), :]
                    #         input_f_ = input_f[int(self.target_h*k): int(self.target_h*(k+1/2)), int(self.target_w*l): int(self.target_w*(l+1/2)), :]
                    #         heatmap_ = heatmap[i][int(self.heatmap_h*k): int(self.heatmap_h*(k+1/2)), int(self.heatmap_w*l): int(self.heatmap_w*(l+1/2))]
                    #         heatmap_f_ = heatmap_f[i][int(self.heatmap_h*k): int(self.heatmap_h*(k+1/2)), int(self.heatmap_w*l): int(self.heatmap_w*(l+1/2))]
                    #         target_ = target[..., 0][int(self.heatmap_h*k): int(self.heatmap_h*(k+1/2)), int(self.heatmap_w*l): int(self.heatmap_w*(l+1/2))]
                    #         target_f_ = target_f[..., 0][int(self.heatmap_h*k): int(self.heatmap_h*(k+1/2)), int(self.heatmap_w*l): int(self.heatmap_w*(l+1/2))]

                    #         # Flipping
                    #         if np.random.random() < 0.5:
                    #             input_ = input_[:, ::-1, :]
                    #             input_f_ = input_f_[:, ::-1, :]
                    #             heatmap_f_ = heatmap_f_[:, ::-1]
                            
                    #         #Normalise + Convert numpy array to tensor
                    #         input_f_ = input_f_/255
                    #         patch_img_trans_f.append(self.final_transforms(input_f_))
                            
                    #         input_ = input_/255
                    #         patch_img_trans.append(self.final_transforms(input_))
                            
                    #         patch_heatmap_r.append(heatmap_)
                    #         patch_heatmap_f.append(heatmap_f_)
                    #         patch_target_r.append(target_/255)
                    #         patch_target_f.append(target_f_/255)
                    # else:
                    patch_img_trans = self.final_transforms(input/255)
                    patch_img_trans_f = self.final_transforms(input_f/255)
                    patch_heatmap_f = heatmap_f
                    patch_heatmap_r = heatmap
                    patch_target_f = target_f/255
                    patch_target_r = target/255
                    patch_cstency_r = cstency_hm/255
                    patch_cstency_f = cstency_hm_f/255
                else:
                    #Normalise + Convert numpy array to tensor
                    img_trans = input/255
                    img_trans = self.final_transforms(img_trans)
                    
                label = np.expand_dims(label, axis=-1)
                flag = False
            except Exception as e:
                # print(f'There is something wrong! Please check the DataLoader!, {e}')
                flag = True
                idx=torch.randint(low=0, high=self.__len__(), size=(1,)).item()
                
        if self.train:
            return patch_img_trans_f, patch_heatmap_f, patch_target_f, patch_cstency_f, patch_img_trans, patch_heatmap_r, patch_target_r, patch_cstency_r
        else:
            return img_trans, label, vid_id
    
    def train_collate_fn(self, batch):
        batch_data = {}
        
        img_f, hm_f, target_f, cst_f, img_r, hm_r, target_r, cst_r = zip(*batch)
        
        #Collating data in case of using spliting images into patches
        # if self.split_image:
        #     hm_H, hm_W = hm_r[0][0].shape
            
        #     img_f = np.reshape(img_f, (-1))
        #     hm_f = np.reshape(hm_f, (-1, 1, hm_H, hm_W))
        #     target_f = np.reshape(target_f, (-1, 1, hm_H, hm_W))
        #     img_r = np.reshape(img_r, (-1))
        #     hm_r = np.reshape(hm_r, (-1, 1, hm_H, hm_W))
        #     target_r = np.reshape(target_r, (-1, 1, hm_H, hm_W))
        
        img = torch.cat([torch.tensor([it.numpy() for it in img_r]), torch.tensor([it.numpy() for it in img_f])], 0)
        heatmap = torch.cat([torch.tensor(hm_r).float(), torch.tensor(hm_f).float()], 0)
        target = torch.cat([torch.tensor(target_r).float(), torch.tensor(target_f).float()], 0)
        label = torch.tensor([[0]] * len(img_r) + [[1]]*len(img_f))
        # label = torch.tensor([0] * len(img_r) + [1]*len(img_f))
        cst = torch.cat([torch.tensor(cst_r).float(), torch.tensor(cst_f).float()], 0)
        
        b_size = label.size(0)
        
        # Permute idxes
        idxes = torch.randperm(b_size)
        img, label, target, heatmap, cst = img[idxes], label[idxes], target[idxes], heatmap[idxes], cst[idxes]
        
        batch_data["img"] = img
        batch_data["label"] = label
        batch_data["target"] = target
        batch_data["heatmap"] = heatmap
        batch_data["cstency_heatmap"] = cst
        
        return batch_data
    
    def train_worker_init_fn(self, worker_id):
        # print('Current state {} --- worker id {}'.format(np.random.get_state()[1][0], worker_id))
        np.random.seed(np.random.get_state()[1][0] + worker_id)


if __name__=="__main__":
    from pipelines.geo_transform import GeometryTransform
    from torch.utils.data import DataLoader
    from configs.get_config import load_config

    PIPELINES.register_module(module=GeometryTransform)
    
    config = load_config("configs/efn4_fpn_sbi_adv.yaml")
    hm_ff = DATASETS.build(cfg=config.DATASET, default_args=dict(split='train', config=config.DATASET))
    hm_ff_loader = DataLoader(hm_ff,
                              batch_size=10,
                              shuffle=True,
                              collate_fn=hm_ff.train_collate_fn,
                              worker_init_fn=hm_ff.train_worker_init_fn)

    for b, batch_data in enumerate(hm_ff_loader):
        inputs, labels, heatmaps, consistencies = batch_data["img"], batch_data["label"], batch_data["heatmap"], batch_data["cstency_heatmap"]
        print(f'X.shape - {inputs.shape}, y shape - {labels.shape}, heatmap shape - {heatmaps.shape}, {heatmaps.max()}, cst shape - {consistencies.shape}')
        break
