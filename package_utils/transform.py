#-*- coding: utf-8 -*-
from copy import deepcopy
import cv2
import numpy as np

from torchvision import transforms
import albumentations as alb


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(center, 
                         scale, 
                         rot, 
                         output_size, 
                         shift=np.array([0, 0], dtype=np.float32), 
                         inv=0, 
                         pixel_std=200):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * pixel_std
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180

    src_dir = get_dir([0, (src_w-1) * -0.5], rot_rad)
    dst_dir = np.array([0, (dst_w-1) * -0.5], np.float32)
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [(dst_w-1) * 0.5, (dst_h-1) * 0.5]
    dst[1, :] = np.array([(dst_w-1) * 0.5, (dst_h-1) * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    """
    This function apply the affine transform to each point given by an affine matrix
    """
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def draw_landmarks(image, landmarks):
    """This function is to draw facial landmarks into transformed images
    """
    assert landmarks is not None, "Landmarks can not be None!"
    
    img_cp = deepcopy(image)
    
    for i, p in enumerate(landmarks):
        img_cp = cv2.circle(img_cp, (p[0], p[1]), 2, (0, 255, 0), 1)
    
    return img_cp


def get_center_scale(shape, aspect_ratio, pixel_std=200):
    h, w = shape[0], shape[1]
    center = np.zeros((2), dtype=np.float32)
    center[0] = (shape[1] - 1) / 2
    center[1] = (shape[0] - 1) / 2
    
    if w > h * aspect_ratio:
        h = w * 1.0 / aspect_ratio
    else:
        w = h * 1.0 / aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
        dtype=np.float32
    )
    
    return center, scale


def final_transform(_cfg):
    return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=_cfg.TRANSFORM.normalize.mean, 
                    std=_cfg.TRANSFORM.normalize.std,
                ),
            ])


def randaffine(img, mask):
    f = alb.Affine(translate_percent={'x':(-0.03,0.03),'y':(-0.015,0.015)},
		           scale=[0.95,1/0.95],
		           fit_output=False,
		           p=1)
			
    g = alb.ElasticTransform(alpha=50, 
                             sigma=7,
		                     alpha_affine=0,
		                     p=1)

    transformed=f(image=img,mask=mask)
    img=transformed['image']
    
    mask=transformed['mask']
    transformed=g(image=img,mask=mask)
    mask=transformed['mask']
    return img, mask
