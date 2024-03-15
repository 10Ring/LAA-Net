#-*- coding: utf-8 -*-
import os
import sys
import simplejson as json
import math
import random

from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from glob import glob
import numpy as np
from box import Box as edict

from .builder import DATASETS
from package_utils.utils import file_extention
from package_utils.transform import final_transform
from package_utils.image_utils import cal_mask_wh, gaussian_radius


PREFIX_PATH = '/data/deepfake_cluster/datasets_df/FaceForensics++/c0/'


@DATASETS.register_module()
class CommonDataset(Dataset, ABC):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self._cfg = edict(cfg) if not isinstance(cfg, edict) else cfg
        self.dataset = self._cfg.DATA[self.split.upper()].NAME
        # self.train = self._cfg["TRAIN"]
        self.train = (self.split != 'test')
        self.final_transforms = final_transform(self._cfg)
        self.sigma_adaptive = self._cfg.ADAPTIVE_SIGMA
        self.sampler_active = self._cfg.DATA.SAMPLES_PER_VIDEO.ACTIVE
        self.samples_per_video = self._cfg.DATA.SAMPLES_PER_VIDEO[self.split.upper()]
        self.heatmap_w = self._cfg.HEATMAP_SIZE[1]
        self.heatmap_h = self._cfg.HEATMAP_SIZE[0]
        self.split_image = self._cfg.SPLIT_IMAGE
        self.compression = self._cfg.COMPRESSION

        if kwargs is not None:
            for k,v in kwargs.items():
                if v is None:
                    raise ValueError(f'{k}:{v} retrieve a None value!')
                self.__setattr__(k, v)

    @abstractmethod
    def _load_from_path(self, split):
        return NotImplemented
    
    def _load_from_file(self, split, anno_file=None):
        """
        @split: train/val
        This function for loading data from file for 4 types of manipulated images FF++ and FaceXray generation data
        """
        assert os.path.exists(self._cfg.DATA[self.split.upper()].ROOT), "Root path to dataset can not be invalid!"
        data_cfg = self._cfg.DATA
        
        if anno_file is None:
            anno_file = data_cfg[split.upper()].ANNO_FILE
        if not os.access(anno_file, os.R_OK):
            anno_file = os.path.join(self._cfg.DATA[self.split.upper()].ROOT, anno_file)
        assert os.access(anno_file, os.R_OK), "Annotation file can not be invalid!!"

        f_name, f_extention = file_extention(anno_file)
        data = None
        image_paths, labels, mask_paths, ot_props = [], [], [], []
        f = open(anno_file)
        if f_extention == '.json':
            data = json.load(f)
            data = edict(data)['data'] #A list of proprocessed data objects containing image properties

            for item in data:
                assert 'image_path' in item.keys(), 'Image path must be available in item dict!'
                image_path = item.image_path
                ot_prop = {}
                
                # Custom base on the specific data structure
                if not 'label' in item.keys():
                    lb = (('fake' in image_path) or (('original' not in image_path) and ('aligned' not in image_path)))
                else:
                    lb = (item.label == 'fake')
                lb_encoded = int(lb)
                labels.append(lb_encoded)
                
                if PREFIX_PATH in item.image_path:
                    image_path = item.image_path.replace(PREFIX_PATH, self._cfg.DATA[self.split.upper()].ROOT)
                else:
                    image_path = os.path.join(self._cfg.DATA[self.split.upper()].ROOT, item.image_path)
                image_paths.append(image_path)

                # Appending more data properties for data loader
                if 'mask_path' in item.keys():
                    mask_path = item.mask_path
                    if PREFIX_PATH in item.mask_path:
                        mask_path = item.mask_path.replace(PREFIX_PATH, self._cfg.DATA[self.split.upper()].ROOT)
                    else:
                        mask_path = os.path.join(self._cfg.DATA[self.split.upper()].ROOT, item.mask_path)
                    mask_paths.append(mask_path)
                if 'best_match' in item.keys():
                    best_match = item.best_match
                    best_match = [os.path.join(self._cfg.DATA[self.split.upper()].ROOT, bm) for bm in best_match if \
                        self._cfg.DATA[self.split.upper()].ROOT not in bm]
                    ot_prop['best_match'] = best_match
                for lms_key in ['aligned_lms', 'orig_lms']:
                    if lms_key in item.keys():
                        f_lms = np.array(item[lms_key])
                        ot_prop[lms_key] = f_lms
                    
                ot_props.append(ot_prop)
        else:
            raise Exception(f'{f_extention} has not been supported yet! Please change to Json file!')
        
        print('{} image paths have been loaded!'.format(len(image_paths)))
        return image_paths, labels, mask_paths, ot_props

    def _encode_target(self, target_mask, normalize_max=True):
        assert self.heatmap_type == 'gaussian', 'Only Gaussian Heatmap is supported now!'
        # tmp = self.sigma * 3
        # size = tmp * 2 + 1
        heatmap = np.zeros((1, target_mask[..., 0].shape[0], target_mask[..., 0].shape[1]), dtype=np.float32)
        hm_w = self._cfg.HEATMAP_SIZE[1]
        hm_h = self._cfg.HEATMAP_SIZE[0]
        
        # Draw heatmap for blending region
        target_mask_ = (target_mask[..., 0] >= 128).astype(np.int8)
        points = np.where(target_mask_ == 1)
        for j, i in zip(points[0], points[1]):
            if self.sigma_adaptive:
                neighbor_list_j = [j, j-1, j + 1]
                neighbor_list_i = [i, i-1, i + 1]
                n_neighbors = 0
                for k in neighbor_list_i:
                    for l in neighbor_list_j:
                        if k in points[1] and l in points[0]:
                            n_neighbors += 1
                self.sigma = 2 * (1/(2 - math.pow(2, 1- n_neighbors)))
            tmp = self.sigma * 3
            size = tmp * 2 + 1
            ul = [int(i - tmp), int(j - tmp)]
            br = [int(i + tmp + 1), int(j + tmp + 1)]
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            
            x0 = y0 = size // 2
            g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (self.sigma ** 2)))
            
            g_x = max(0, -ul[0]), min(br[0], hm_w) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], hm_h) - ul[1]
            
            img_x = max(0, ul[0]), min(br[0], hm_w)
            img_y = max(0, ul[1]), min(br[1], hm_h)
            
            heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] += g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        
        # Normalize heatmap values
        if normalize_max:
            hm_max_val = heatmap.max()
            if hm_max_val > 0:
                heatmap = heatmap/hm_max_val
        else:
            hm_cx = hm_w//2
            hm_cy = hm_h//2
            
            heatmap[0:hm_cy, 0:hm_cx] = heatmap[0:hm_cy, 0:hm_cx]/(heatmap[0:hm_cy, 0:hm_cx].max() + 1e-6)
            heatmap[0:hm_cy, hm_cx:hm_w] = heatmap[0:hm_cy, hm_cx:hm_w]/(heatmap[0:hm_cy, hm_cx:hm_h].max() + 1e-6)
            heatmap[hm_cy:hm_h, 0:hm_cx] = heatmap[hm_cy:hm_h, 0:hm_cx]/(heatmap[hm_cy:hm_h, 0:hm_cx].max() + 1e-6)
            heatmap[hm_cy:hm_h, hm_cx:hm_w] = heatmap[hm_cy:hm_h, hm_cx:hm_w]/(heatmap[hm_cy:hm_h, hm_cx:hm_w].max() + 1e-6)
        return heatmap, None
    
    def _new_encode_target(self, target_mask, normalize_max=True):
        assert self.heatmap_type == 'gaussian', 'Only Gaussian Heatmap is supported now!'
        # fake_ratio = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        n_outputs = 1
        hm_w = self._cfg.HEATMAP_SIZE[1]
        hm_h = self._cfg.HEATMAP_SIZE[0]
        patches = [[0, 0], [0, 1/2], [1/2, 0], [1/2, 1/2]]
        target_H, target_W = target_mask[..., 0].shape[:2]
        heatmap = np.zeros((n_outputs, target_H, target_W), dtype=np.float32)
        cstency_hm = np.zeros((n_outputs, target_H, target_W), dtype=np.float32)
        max_val_all = target_mask[..., 0].max()
        
        # Draw heatmap for blending region
        for fr in range(len(patches)):
            # target_mask_ = np.where(((target_mask[..., 0] > 255*fake_ratio[fr]) & (target_mask[..., 0] <= 255*fake_ratio[fr+1])), 1, 0)
            p_x1, p_y1 = int(target_W * patches[fr][0]), int(target_H * patches[fr][1])
            p_x2, p_y2 = int(target_W * (patches[fr][0] + 1/2)), int(target_H * (patches[fr][1] + 1/2))

            max_value = target_mask[p_y1:p_y2, p_x1:p_x2, 0].max()
            max_value = max_value if max_value > 0 else 1
            target_mask_ = (target_mask[p_y1:p_y2, p_x1:p_x2, 0] == (max_value)).astype(np.uint8)
            points = np.where(target_mask_ == 1)
            
            if len(points[0]):
                p = (points[0] + p_y1, points[1] + p_x1)

                for j, i in zip(p[0], p[1]):
                    if self.sigma_adaptive:
                        w_sbi, h_sbi = cal_mask_wh((j, i), target_mask[..., 0])
                        radius = gaussian_radius((h_sbi, w_sbi))
                        self.sigma = radius/3 + 1e-4
                    tmp = self.sigma * 3
                    size = tmp * 2 + 1
                    ul = [int(i - tmp), int(j - tmp)]
                    br = [int(i + tmp + 1), int(j + tmp + 1)]
                    x = np.arange(0, size, 1, np.float32)
                    y = x[:, np.newaxis]
                    
                    x0 = y0 = size // 2
                    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (self.sigma ** 2)))
                    
                    g_x = max(0, -ul[0]), min(br[0], hm_w) - ul[0]
                    g_y = max(0, -ul[1]), min(br[1], hm_h) - ul[1]
                    
                    img_x = max(0, ul[0]), min(br[0], hm_w)
                    img_y = max(0, ul[1]), min(br[1], hm_h)
                    
                    heatmap[0][img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]],
                        heatmap[0][img_y[0]:img_y[1], img_x[0]:img_x[1]])
                
                if n_outputs > 1:
                    cstency_hm[fr][p_y1:p_y2, p_x1:p_x2] = 255 - np.absolute(max_value - target_mask[p_y1:p_y2, p_x1:p_x2, 0])
                else:
                    cstency_hm[0][p_y1:p_y2, p_x1:p_x2] = 255 - np.absolute(max_val_all - target_mask[p_y1:p_y2, p_x1:p_x2, 0])
                
        return heatmap, cstency_hm
    
    def _encode_target_v2(self, target_mask, normalize_max=True):
        assert self.heatmap_type == 'gaussian', 'Only Gaussian Heatmap is supported now!'
        n_outputs = 1
        hm_w = self._cfg.HEATMAP_SIZE[1]
        hm_h = self._cfg.HEATMAP_SIZE[0]
        target_H, target_W = target_mask[..., 0].shape[:2]
        heatmap = np.zeros((n_outputs, target_H, target_W), dtype=np.float32)
        cstency_hm = np.zeros((n_outputs, target_H, target_W), dtype=np.float32)
        
        # Draw heatmap for blending region
        target_mask_ = (target_mask[..., 0] > 128).astype(np.uint8)
        points = np.where(target_mask_ == 1)
            
        if len(points[0]):
            p = (int(points[0].mean()), int(points[1].mean()))
            j,i = p
            
            if self.sigma_adaptive:
                w_sbi, h_sbi = cal_mask_wh((j, i), target_mask[..., 0])
                radius = gaussian_radius((h_sbi, w_sbi))
                self.sigma = radius/3 + 1e-4
            tmp = self.sigma * 3
            size = tmp * 2 + 1
            ul = [int(i - tmp), int(j - tmp)]
            br = [int(i + tmp + 1), int(j + tmp + 1)]
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            
            x0 = y0 = size // 2
            g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (self.sigma ** 2)))
            
            g_x = max(0, -ul[0]), min(br[0], hm_w) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], hm_h) - ul[1]
            
            img_x = max(0, ul[0]), min(br[0], hm_w)
            img_y = max(0, ul[1]), min(br[1], hm_h)
            
            heatmap[0][img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]],
                heatmap[0][img_y[0]:img_y[1], img_x[0]:img_x[1]])

            cstency_hm[0] = 255 - np.absolute(target_mask[j,i,0] - target_mask[..., 0])
            
        return heatmap, cstency_hm

    def _sampler(self, image_paths, labels, **params):
        vid_dict = {}
        data = {"image_paths": [], "labels": []}
        
        for k,v in params.items():
            if v is not None and len(v): data[k] = []
        
        for idx, ip in enumerate(image_paths):
            f_name = ip.split('/')[-1]
            if self.compression == 'c23':
                vid_id = f_name.split('_frame')[0]
            elif self.compression == 'c0':
                vid_id = ip.split('/')[-2]
            else:
                raise NotImplementedError('Only c23 and c0 compression mode is supported now! Please check again!')
            lb = labels[idx]
            
            data_per_vid = dict(image=ip, label=lb)
            for k,v in params.items():
                if k in data.keys():
                    data_per_vid[k] = v[idx]
            
            if vid_id in vid_dict.keys():
                vid_dict[vid_id].append(data_per_vid)
            else:
                vid_dict[vid_id] = [data_per_vid]
                
        for vid_id in vid_dict.keys():
            samples_per_vid = random.choices(vid_dict[vid_id], k=self.samples_per_video)
            for spl in samples_per_vid:
                data["image_paths"].append(spl["image"])
                data["labels"].append(spl["label"])
                for k in params.keys():
                    if k in data.keys():
                        data[k].append(spl[k])
        return data
    
    def select_encode_method(self, version=0):
        if version==2:
            return self._encode_target_v2
        elif version==1:
            return self._new_encode_target
        else:
            return self._encode_target
    
    @abstractmethod
    def __len__(self):
        return NotImplemented

    @abstractmethod
    def __getitem__(self, idx):
        return NotImplemented

    @property
    def __repr__(self):
        return self.__class__.__name__
