#-*- coding: utf-8 -*-
import os
import sys
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

import albumentations as alb
from albumentations.core.transforms_interface import ImageOnlyTransform
import cv2
import numpy as np
from imgaug import augmenters as iaa

from package_utils.deepfake_mask import dynamic_blend, random_get_hull
from package_utils.transform import randaffine
from package_utils.image_utils import load_image
from package_utils.bi_online_generation import random_erode_dilate


# predefine mask distortion
distortion = iaa.Sequential([iaa.PiecewiseAffine(scale=(0.01, 0.15))])


class RandomDownScale(ImageOnlyTransform):
    def apply(self, img, **params):
        return self.randomdownscale(img)
    
    def randomdownscale(self, img):
        keep_ratio = True
        keep_input_shape = True
        H, W, C = img.shape
        ratio_list = [2,4]
        r = ratio_list[np.random.randint(len(ratio_list))]
        # r = np.random.uniform(2, 4)
        img_ds = cv2.resize(img, (int(W/r), int(H/r)), interpolation=cv2.INTER_NEAREST)
        if keep_input_shape:
            img_ds = cv2.resize(img_ds, (W,H), interpolation=cv2.INTER_LINEAR)
            
        return img_ds


def get_source_transforms():
    return alb.Compose([
        alb.Compose([
            alb.RGBShift((-20,20), (-20,20), (-20,20), p=0.3),
            alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=1),
            alb.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1,0.1), p=1),
        ],p=1),

        alb.OneOf([
            RandomDownScale(p=1),
            alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
        ],p=1),
    ], p=1.)


def get_transforms():
    return alb.Compose([
            alb.RGBShift((-20,20), (-20,20), (-20,20), p=0.3),
            alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=0.3),
            alb.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3,0.3), p=0.3),
            alb.ImageCompression(quality_lower=40, quality_upper=100, p=0.5),
        ], 
        additional_targets={f'image_f': 'image'},
        p=1.)


def gen_SBI(img, landmark):
    H, W = len(img), len(img[0])
    distort = False
    if np.random.rand()<0.25:
        distort = True
        landmark = landmark[:68]
    
    # Getting ConvexHull
    mask = random_get_hull(landmark, img)[:,:,0]

    source = img.copy()
    if np.random.rand() < 0.5:
        source = get_source_transforms()(image=source.astype(np.uint8))['image']
    else:
        img = get_source_transforms()(image=img.astype(np.uint8))['image']

    # Mask Deformation for diverse blending
    # if not distort:
    source, mask = randaffine(source, mask)
    # else:
    #     mask = distortion.augment_image(mask)
    #     mask = random_erode_dilate(mask)

    img_blended, mask = dynamic_blend(source, img, mask)
    img_blended = img_blended.astype(np.uint8)
    img = img.astype(np.uint8)

    return img, img_blended, mask


def gen_target(background_face, background_landmark, margin=[20, 20]):
    if isinstance(background_face, str):
        background_face = load_image(background_face)
    
    margin_x, margin_y = margin

    background_face, face_img, mask_f = gen_SBI(background_face, background_landmark)
    mask_f = (1 - mask_f) * mask_f * 4
    mask_r = np.zeros((mask_f.shape[0], mask_f.shape[1], 1))
    
    H, W = len(face_img), len(face_img[0])
    face_img = face_img[margin_y:(H-margin_y), margin_x:(W-margin_x), :]
    background_face = background_face[margin_y:(H-margin_y), margin_x:(W-margin_x), :]
    
    mask_f = mask_f[margin_y:(H-margin_y), margin_x:(W-margin_x), :]
    mask_r = mask_r[margin_y:(H-margin_y), margin_x:(W-margin_x), :]
    
    mask_f, mask_r = np.repeat(mask_f,3,2), np.repeat(mask_r,3,2)
    mask_f, mask_r = (mask_f*255).astype(np.uint8), (mask_r*255).astype(np.uint8)
    return face_img, mask_f, background_face, mask_r


def reorder_landmark(landmark):
    landmark_add=np.zeros((13,2))
    for idx,idx_l in enumerate([77,75,76,68,69,70,71,80,72,73,79,74,78]):
        landmark_add[idx]=landmark[idx_l]
    landmark[68:]=landmark_add
    return landmark


def sbi_hflip(img, mask=None, landmark=None, bbox=None):
    H, W = img.shape[:2]
    if landmark is not None:
        landmark = landmark.copy()

    if bbox is not None:
        bbox = bbox.copy()

    if landmark is not None:
        landmark_new=np.zeros_like(landmark)

        landmark_new[:17]=landmark[:17][::-1]
        landmark_new[17:27]=landmark[17:27][::-1]

        landmark_new[27:31]=landmark[27:31]
        landmark_new[31:36]=landmark[31:36][::-1]

        landmark_new[36:40]=landmark[42:46][::-1]
        landmark_new[40:42]=landmark[46:48][::-1]

        landmark_new[42:46]=landmark[36:40][::-1]
        landmark_new[46:48]=landmark[40:42][::-1]

        landmark_new[48:55]=landmark[48:55][::-1]
        landmark_new[55:60]=landmark[55:60][::-1]

        landmark_new[60:65]=landmark[60:65][::-1]
        landmark_new[65:68]=landmark[65:68][::-1]
        if len(landmark)==68:
            pass
        elif len(landmark)==81:
            landmark_new[68:81]=landmark[68:81][::-1]
        else:
            raise NotImplementedError
        landmark_new[:,0]=W-landmark_new[:,0]
    else:
        landmark_new=None

    if bbox is not None:
        bbox_new=np.zeros_like(bbox)
        bbox_new[0,0]=bbox[1,0]
        bbox_new[1,0]=bbox[0,0]
        bbox_new[:,0]=W-bbox_new[:,0]
        bbox_new[:,1]=bbox[:,1].copy()
        if len(bbox)>2:
            bbox_new[2,0]=W-bbox[3,0]
            bbox_new[2,1]=bbox[3,1]
            bbox_new[3,0]=W-bbox[2,0]
            bbox_new[3,1]=bbox[2,1]
            bbox_new[4,0]=W-bbox[4,0]
            bbox_new[4,1]=bbox[4,1]
            bbox_new[5,0]=W-bbox[6,0]
            bbox_new[5,1]=bbox[6,1]
            bbox_new[6,0]=W-bbox[5,0]
            bbox_new[6,1]=bbox[5,1]
    else:
        bbox_new=None

    if mask is not None:
        mask=mask[:,::-1]
    else:
        mask=None
    img=img[:,::-1].copy()
    return img,mask,landmark_new,bbox_new
