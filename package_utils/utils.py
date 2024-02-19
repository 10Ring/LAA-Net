#-*- coding: utf-8 -*-
import os
import simplejson as json

import cv2
import numpy as np
import torch
from losses.losses import _sigmoid


def file_extention(file_path):
    f_name, f_extension = os.path.splitext(file_path)
    return f_name, f_extension


def make_dir(dir_path):
    if not os.path.exists(dir_path):
       os.mkdir(dir_path)


def vis_heatmap(image, heatmaps, file_name):
    hm_h, hm_w = heatmaps.shape[1:]
    masked_image = np.zeros((hm_h*heatmaps.shape[0], hm_w, 3))

    for i in range(heatmaps.shape[0]):
        heatmap = heatmaps[i]
        heatmap = np.clip(heatmap*255, 0, 255).astype(np.uint8)
        heatmap = np.squeeze(heatmap)
        
        heatmap_h = heatmap.shape[0]
        heatmap_w = heatmap.shape[1]
        
        resized_image = cv2.resize(image, (int(heatmap_h), int(heatmap_w)))
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        masked_image[hm_h*i:hm_h*(i+1), :] = colored_heatmap*0.7 + resized_image*0.3
    cv2.imwrite(file_name, masked_image)


def save_batch_heatmaps(batch_image, 
                        batch_heatmaps, 
                        file_name,
                        normalize=True,
                        batch_cls=None):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    batch_cls: ['batch_size, num_joints, 1]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)
    
    grid_image = np.zeros((batch_size*heatmap_height,
                           (num_joints+1)*heatmap_width,
                           3),
                          dtype=np.uint8)

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            if batch_cls is not None:
                cls = batch_cls[i][j].detach().cpu().numpy()
                colored_heatmap = cv2.putText(colored_heatmap, 
                                              f'Cls Pred: {cls}', 
                                              (heatmap_width*(j+1)-15, 10), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 
                                              1, 1, 
                                              cv2.LINE_AA)
            masked_image = colored_heatmap*0.7 + resized_image*0.3

            width_begin = heatmap_width * (j+1)
            width_end = heatmap_width * (j+2)
            
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image
    cv2.imwrite(file_name, grid_image)


def debugging_panel(debug_cfg, 
                    batch_image, 
                    batch_heatmaps_gt, 
                    batch_heatmaps_pred, 
                    idx, 
                    normalize=True,
                    batch_cls_gt=None,
                    batch_cls_pred=None,
                    split='train'):
    if debug_cfg.save_hm_gt:
        save_batch_heatmaps(batch_image, 
                            batch_heatmaps_gt, 
                            f'samples/{split}_debugs/hm_gt_{idx}.jpg', 
                            normalize=normalize)
    
    if debug_cfg.save_hm_pred:
        batch_heatmaps_pred_ = _sigmoid(batch_heatmaps_pred.clone())
        save_batch_heatmaps(batch_image, 
                            batch_heatmaps_pred_, 
                            f'samples/{split}_debugs/hm_pred_{idx}.jpg',
                            normalize=normalize)


def save_file(data, file_path):
    f_name, f_extention = file_extention(file_path)
    
    if f_extention == '.json':
        with open(file_path, 'w') as f:
            json.dump(data, f)
        print(f'Data has been saved to --- {file_path}')
    else:
        raise ValueError(f'{f_extention} is not supported now!')


def load_file(file_path):
    f_name, f_extention = file_extention(file_path)
    
    if f_extention == '.json':
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f'Data has been loaded from --- {file_path}')
    else:
        raise ValueError(f'{f_extention} is not supported now!')

    return data
