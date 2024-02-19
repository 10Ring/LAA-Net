#-*- coding: utf-8 -*-
import random

from PIL import Image
import numpy as np
import cv2
from skimage import transform as sktransform
from imgaug import augmenters as iaa
import torch

from .builder import DATASETS, PIPELINES, build_pipeline
from .master import MasterDataset
from package_utils.transform import get_affine_transform, get_center_scale
from package_utils.utils import vis_heatmap
from package_utils.image_utils import load_image
from package_utils.bi_online_generation import (
    random_get_hull, colorTransfer, blendImages, random_erode_dilate
)


@DATASETS.register_module()
class HeatmapFaceForensic(MasterDataset):
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
        super().__init__(config, **kwargs)
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
        self.image_paths, self.labels, self.mask_paths, self.ot_props = self._load_data(split)
        
        # predefine mask distortion
        self.distortion = iaa.Sequential([iaa.PiecewiseAffine(scale=(0.01, 0.15))])
        
        #Calling transform methods for inputs
        self.geo_transform = build_pipeline(config.TRANSFORM.geometry, PIPELINES)
        self.colorjitter_transform = build_pipeline(config.TRANSFORM.color, PIPELINES)

    def _load_data(self, split):
        from_file = self._cfg.DATA[self.split.upper()].FROM_FILE
        
        if not from_file:
            image_paths, labels, mask_paths, ot_props = self._load_from_path(split)
        else:
            image_paths, labels, mask_paths, ot_props = self._load_from_file(split)
            
        if self.sampler_active and self.train:
            print('Running sampler...')
            params = dict(mask_paths=mask_paths, ot_props=ot_props)
            data_sampler = self._sampler(image_paths, labels, **params)
            image_paths, labels = data_sampler['image_paths'], data_sampler['labels']
            if len(mask_paths):
                mask_paths = data_sampler['mask_paths']
            if len(ot_props):
                ot_props = data_sampler['ot_props']
            print(f'n samples after running sampling --- {len(image_paths)}')
            
        assert len(image_paths) != 0, "Image paths have not been loaded! Please check image directory!"
        assert len(labels) != 0, "Labels have not been loaded! Please check annotation file!"
        # if not self.dynamic_fxray or self.split == 'val':
        if from_file and not (self.dynamic_fxray):
            assert len(mask_paths) != 0, "Mask paths have not been loaded! Please check mask directory!"
        return image_paths, labels, mask_paths, ot_props
    
    def _reload_data(self):
        self.image_paths, self.labels, self.mask_paths, self.ot_props = self._load_data(self.split)
        
    def _gen_BI(self, background_face, background_landmark, foreground_face_path, idx=None):
        foreground_face = load_image(foreground_face_path)
        
        # down sample before blending
        aug_size = random.randint(128, 317)
        background_landmark = background_landmark * (aug_size/317)
        foreground_face = sktransform.resize(foreground_face,(aug_size,aug_size),preserve_range=True).astype(np.uint8)
        background_face = sktransform.resize(background_face,(aug_size,aug_size),preserve_range=True).astype(np.uint8)
        
        # get random type of initial blending mask
        mask = random_get_hull(background_landmark, background_face)
        
        if self.debug:
            Image.fromarray((mask*255).astype(np.uint8)).save(f'samples/debugs/orig_CH_{idx}.jpg')
        
        #  random deform mask
        mask = self.distortion.augment_image(mask)
        mask = random_erode_dilate(mask)
        
        if self.debug:
            Image.fromarray((mask*255).astype(np.uint8)).save(f'samples/debugs/deformed_CH_{idx}.jpg')
        
        # filte empty mask after deformation
        if np.sum(mask) == 0 :
            raise NotImplementedError
        
        # apply color transfer
        foreground_face = colorTransfer(background_face, foreground_face, mask*255)
        
        # blend two face
        blended_face, mask = blendImages(foreground_face, background_face, mask*255)
        blended_face = blended_face.astype(np.uint8)
        
        # resize back to default resolution
        blended_face = sktransform.resize(blended_face, (317, 317), preserve_range=True).astype(np.uint8)
        mask = sktransform.resize(mask, (317, 317), preserve_range=True)
        mask = mask[:,:,0:1]
        return blended_face, mask
    
    def _gen_target(self, background_face, background_landmark, foreground_face_path, idx=None):
        data_type = 'real' if random.randint(0,1) else 'fake'
        
        if not background_landmark.any():
            data_type = 'real'
        
        if data_type == 'fake' :
            face_img, mask =  self._gen_BI(background_face, background_landmark, foreground_face_path, idx=idx)
            mask = (1 - mask) * mask * 4
        else:
            face_img = background_face
            mask = np.zeros((317, 317, 1))
        
        face_img = Image.fromarray(face_img)
        # randomly downsample after BI pipeline
        if random.randint(0,1):
            aug_size = random.randint(64, 317)
            if random.randint(0,1):
                face_img = face_img.resize((aug_size, aug_size), Image.BILINEAR)
            else:
                face_img = face_img.resize((aug_size, aug_size), Image.NEAREST)
        face_img = face_img.resize((317, 317),Image.BILINEAR)
        face_img = np.array(face_img)
        
        face_img = face_img[60:(317), 30:(287), :]
        mask = mask[60:(317), 30:(287), :]
        mask = np.repeat(mask,3,2)
        mask = (mask*255).astype(np.uint8)
        return face_img, mask, int(data_type == 'fake')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        flag = True
        while flag:
            try:
                img_path = self.image_paths[idx]
                label = self.labels[idx]
                vid_id = img_path.split('/')[-2]
                img = load_image(img_path)
                mask = None

                if self.debug:
                    Image.fromarray(img).save(f'samples/debugs/orig_{idx}.jpg')
                
                #Applying color transform to inputs
                if self.split == 'train':
                    color_transfomed = self.colorjitter_transform(img)
                    img = color_transfomed['image']
                
                # if not self.dynamic_fxray or self.split == 'val':
                if not self.dynamic_fxray:
                    if bool(self.mask_paths):
                        mask_path = self.mask_paths[idx]
                        mask = load_image(mask_path)
                else:
                    if self.train:
                        best_match = self.ot_props[idx]['best_match'] if len(self.ot_props[idx]['best_match']) else []
                        if len(self.ot_props[idx]['aligned_lms']):
                            f_lms = self.ot_props[idx]['aligned_lms']
                        elif len(self.ot_props[idx]['orig_lms']):
                            f_lms = self.ot_props[idx]['orig_lms']
                        else:
                            f_lms = []
                            
                        if self.debug:
                            img_lms_draw = draw_landmarks(img, f_lms)
                            Image.fromarray(img_lms_draw).save(f'samples/debugs/orig_{idx}_lms.jpg')
                        
                        if len(best_match):
                            best_match_idx = random.randint(0, len(best_match)-1)
                            best_match_path = best_match[best_match_idx]
                            img, mask, label = self._gen_target(img, f_lms, best_match_path, idx=idx)
                        else:
                            img, mask, label = self._gen_target(img, np.array([]), '')
                    else:
                        img = cv2.resize(img, (317,317))
                        img = img[18:(299), 18:(299), :]
                target = None
                
                if mask is not None:
                    assert (mask.shape[:2] == img.shape[:2]), "Color Image and Mask must have the same shape!"
                
                #Applying geo transform to inputs and masks
                if self.split == 'train':
                    geo_transfomed = self.geo_transform(img, mask=mask)
                    img = geo_transfomed['image']
                    mask = geo_transfomed['mask']
                
                #Applying affine transform
                c, s = get_center_scale(img.shape[:2], self.aspect_ratio, pixel_std=self.pixel_std)
                trans = get_affine_transform(c, s, self.rot, self._cfg.IMAGE_SIZE)
                trans_heatmap = get_affine_transform(c, s, self.rot, self._cfg.HEATMAP_SIZE)
                
                input = cv2.warpAffine(img,
                                    trans,
                                    (int(self._cfg.IMAGE_SIZE[0]), int(self._cfg.IMAGE_SIZE[1])),
                                    flags=cv2.INTER_LINEAR)
                
                if mask is not None:
                    target = cv2.warpAffine(mask,
                                            trans_heatmap,
                                            (int(self._cfg.HEATMAP_SIZE[0]), int(self._cfg.HEATMAP_SIZE[1])),
                                            flags=cv2.INTER_LINEAR)

                #Target encoding
                #0 for original, 1 for FXRay, 2 for NoFXRay. If 2, comment: mask = (1 - mask) * mask * 4
                heatmap, cstency_hm = self.select_encode_method(version=1)(target) if (target is not None and self.heatmap_type=='gaussian' and self.train) else (None, None)
                
                if self.debug:
                    Image.fromarray(input).save(f'samples/debugs/affine_{idx}.jpg')
                    Image.fromarray(target).save(f'samples/debugs/mask_affine_{idx}.jpg')
                    vis_heatmap(input, cstency_hm/255, f'samples/debugs/cstency_mask_{idx}.jpg')
                    vis_heatmap(input, heatmap, f'samples/debugs/hm_{idx}.jpg')

                if self.train:
                    if self.split_image:
                        patch_img_trans = []
                        patch_heatmap = []
                        patch_cstency_hm = []
                        patch_target = []
                        patch_label = np.expand_dims(np.tile(label, len(heatmap)), -1)
                        
                        for i, (k, l) in enumerate([[0,0], [1/2,0], [0,1/2], [1/2,1/2]]):
                            input_ = input[int(self.target_h*k): int(self.target_h*(k+1/2)), int(self.target_w*l): int(self.target_w*(l+1/2)), :]
                            heatmap_ = heatmap[i][int(self.heatmap_h*k): int(self.heatmap_h*(k+1/2)), int(self.heatmap_w*l): int(self.heatmap_w*(l+1/2))]
                            cstency_ = cstency_hm[i][int(self.heatmap_h*k): int(self.heatmap_h*(k+1/2)), int(self.heatmap_w*l): int(self.heatmap_w*(l+1/2))]
                            target_ = target[..., 0][int(self.heatmap_h*k): int(self.heatmap_h*(k+1/2)), int(self.heatmap_w*l): int(self.heatmap_w*(l+1/2))]
                            
                            #Normalise + Convert numpy array to tensor
                            input_ = input_/255
                            patch_img_trans.append(self.final_transforms(input_))
                            
                            patch_heatmap.append(heatmap_)
                            patch_cstency_hm.append(cstency_/255)
                            patch_target.append(target_/255)
                    else:
                        patch_img_trans = self.final_transforms(input/255)
                        patch_heatmap = heatmap
                        patch_cstency_hm = cstency_hm/255
                        patch_target = target/255
                        patch_label = np.expand_dims(label, axis=-1)
                        # patch_label = label
                else:
                    #Normalise + Convert numpy array to tensor
                    img_trans = input/255
                    img_trans = self.final_transforms(img_trans)
                    label = np.expand_dims(label, axis=-1)
                flag = False
            except Exception as e:
                print('There is an exception during loading data, please check --- ', e)
                idx=torch.randint(low=0, high=self.__len__(), size=(1,)).item()
        
        if self.train:
            return patch_img_trans, patch_label, patch_target, patch_heatmap, patch_cstency_hm
        else:
            return img_trans, label, vid_id

    def train_collate_fn(self, batch):
        batch_data = {}
        img, label, target, hm, cstency_hm = zip(*batch)
        
        #Collating data in case of using spliting images into patches
        if self.split_image:
            hm_H, hm_W = hm[0][0].shape
            
            img = np.reshape(img, (-1))
            hm = np.reshape(hm, (-1, 1, hm_H, hm_W))
            cstency_hm = np.reshape(cstency_hm, (-1, 1, hm_H, hm_W))
            target = np.reshape(target, (-1, 1, hm_H, hm_W))
            label = np.reshape(label, (-1, 1))
        
        img = torch.tensor([it.numpy() for it in img])
        heatmap = torch.tensor(hm).float()
        cstency_heatmap = torch.tensor(cstency_hm).float()
        target = torch.tensor(target).float()
        label = torch.tensor(label)
        
        batch_data["img"] = img
        batch_data["label"] = label
        batch_data["target"] = target
        batch_data["heatmap"] = heatmap
        batch_data["cstency_heatmap"] = cstency_heatmap
        
        return batch_data

if __name__=="__main__":
    # from datasets import *
    from pipelines.geo_transform import GeometryTransform
    from pipelines.color_transform import ColorJitterTransform
    from torch.utils.data import DataLoader
    from configs.get_config import load_config
    
    PIPELINES.register_module(module=GeometryTransform)
    PIPELINES.register_module(module=ColorJitterTransform)

    config = load_config("configs/efn4_fpn_hm.yaml")
    hm_ff = DATASETS.build(cfg=config.DATASET, default_args=dict(split='train', config=config.DATASET))
    hm_ff_loader = DataLoader(hm_ff,
                              batch_size=10,
                              shuffle=False,
                              collate_fn=hm_ff.train_collate_fn)
    for b, batch_data in enumerate(hm_ff_loader):
        inputs, labels, targets, heatmaps, cstency_heatmap = \
            batch_data["img"], batch_data['label'], batch_data['target'], batch_data['heatmap'], batch_data['cstency_heatmap']
        print(f'X.shape - {inputs.shape}, y shape - {labels.shape}, heatmaps - {heatmaps.shape}, consistency -- {cstency_heatmap.shape}')
        break
