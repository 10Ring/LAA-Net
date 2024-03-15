#-*- coding: utf-8 -*-
import math
import os
import sys
import argparse
import time
if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())
import random

import simplejson as json
from box import Box as edict
from glob import glob
import dlib
import cv2
import numpy as np
from imutils import face_utils
from tqdm import tqdm

from configs.get_config import load_config
from transform import affine_transform, draw_landmarks


class LandmarkUtility(object):
    def __init__(self,
                 cfg,
                 load_imgs=False,                 
                 **kwargs):
        super().__init__()

        assert "DATASET" in cfg, "Dataset can not be None!"
        assert "ROOT" in cfg, "Image Directory need to be provided!"

        if not isinstance(cfg, edict):
            cfg = edict(cfg)

        self.load_imgs = load_imgs
        self.image_root = cfg.ROOT
        self.image_suffix = cfg.IMAGE_SUFFIX or 'jpg'
        self.dataset = cfg.DATASET
        self.split = cfg.SPLIT or 'train'
        self.data_type = cfg.DATA_TYPE or 'images'
        self.fol_label = cfg.LABEL or ['real']
        self.debug = cfg.DEBUG
        self.fake_types = cfg.FAKETYPE
        self.compression = cfg.COMPRESSION

        if kwargs is not None:
            for k,v in kwargs.items():
                if v is None:
                    raise ValueError(f'{k}:{v} recieve a None value!')
                self.__setattr__(k, v)

    def __contain__(self, key):
        return hasattr(self, key)

    def _load_data(self):
        img_paths = []
        file_names = []

        print(f'Loading data from dataset --- {self.dataset}')
        if self.load_imgs:
            img_paths, file_names = self._load_data_from_path()
        else:
            assert self.__contain__('file_path'), "Loading data from file need a file path"
            img_paths, file_names = self._load_data_from_file(self.__getattribute__('file_path'))

        assert len(img_paths) != 0, "Image paths have not been loaded! Please check image directory!"
        assert len(file_names) != 0, "Image files have not been loaded! Please check image suffixes!"
        return img_paths, file_names
    
    def _load_data_from_path(self):
        """
        Currenly, Using Glob for loading file with regex
        It might be changed for better performance in large datasets
        """
        assert os.path.exists(self.image_root), "Root path to dataset can not be None!"
        data_type = self.data_type
        fake_types = self.fake_types
        img_paths = []

        # Load image data for each type of fake techniques
        for idx, ft in enumerate(fake_types):
            data_dir = os.path.join(self.image_root, self.split, data_type, ft)
            if not os.path.exists(data_dir):
                raise ValueError("Data Directory can not be invalid!")
            
            for sub_dir in os.listdir(data_dir):
                sub_dir_path = os.path.join(data_dir, sub_dir)
                img_paths_ = glob(f'{sub_dir_path}/*.{self.image_suffix}')

                img_paths.extend(img_paths_)
                
        print('{} image paths have been loaded from {}!'.format(len(img_paths), self.dataset))
        file_names = [ip.split('/')[-1] for ip in img_paths]

        return img_paths, file_names

    def _load_data_from_file(self, file_path):
        """
        Each extension will be treated with particular extension loader
        """
        filename, file_extension = os.path.splitext(file_path)
        img_paths, file_names = [], []
        if file_extension == '.json':
            f = open(file_path)
            data = json.load(f)
            obj_data = data["data"]
            
            for item in obj_data:
                img_paths.append(item['image_path'])
                file_names.append(item['file_name'])
        return img_paths, file_names

    def _img_obj(self, img_path, file_name, **kwargs):
        obj = dict(image_path=img_path, file_name=file_name, **kwargs)
        return obj
    
    def _load_image(self, img_path):
        image = cv2.imread(os.path.join(self.image_root, img_path))
        return image

    def _facial_landmark(self, image, detector, lm_predictor):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except:
            gray = image
        
        f_rect = detector(gray, 1)
        if len(f_rect) > 0:
            f_lms = lm_predictor(gray, f_rect[0])
            f_lms = face_utils.shape_to_np(f_lms)
            return f_lms
        else:
            return None
        
    def _align_face(self, image, f_lms):
        assert f_lms is not None, "Facial Landmarks can not be None!"
        eyepoints = f_lms[39], f_lms[42]
        le_x, le_y = eyepoints[0]
        re_x, re_y = eyepoints[1]
        
        angle = math.atan((le_y - re_y)/(le_x - re_x)) * (180/math.pi)
        origin_point = tuple(np.array(image.shape[1::-1]) / 2)
        
        rot_mat = cv2.getRotationMatrix2D(origin_point, angle, 1.0)
        rot_img = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        
        # Aligning face landmarks by the rotation matrix
        rot_f_lms = None
        if f_lms is not None:
            rot_f_lms = np.empty_like(f_lms)
            for i, p in enumerate(f_lms):
                rot_f_lms[i] = affine_transform(p, rot_mat)
            
        return rot_img, f_lms, rot_f_lms
        
    def facial_landmarks(self, img_paths, detector, lm_predictor):
        rot_imgs, f_lmses, rot_f_lmses = [], [], [] 
        
        for i, ip in enumerate(tqdm(img_paths, dynamic_ncols=True)):
            image = self._load_image(ip)
            
            # Checking time processing for each item
            s_t = time.time()
            f_lms = None
            try:
                f_lms = self._facial_landmark(image, detector, lm_predictor)
                if f_lms is None:
                    if self.debug:
                        cv2.imwrite(f'samples/exception_img_{i}.jpg', image)
                    print(f'Image {i}--{ip} did not find any landmarks!')
            except Exception as e:
                print(e)
                
            if i == 1:
                print('Landmark detection processing time ---- {}'.format(time.time() - s_t))
            
            if (f_lms is not None):
                rot_img, _f_lms, rot_f_lms = self._align_face(image, f_lms)  
            else: 
                rot_img, _f_lms, rot_f_lms = image, [], []
            rot_imgs.append(rot_img)
            f_lmses.append(_f_lms)
            rot_f_lmses.append(rot_f_lms)
            
            # Visualizing landmarks to test
            if i < 10 and self.debug:
                rot_img = draw_landmarks(rot_img, rot_f_lms)
                cv2.imwrite(f'samples/test_{i}.jpg', rot_img)
            
            if i % 100 == 0:
                print(f'Landmarks have been detected for {i} images')
        return rot_imgs, f_lmses, rot_f_lmses

    def build_data(self, img_paths, file_names, **kwargs):
        data = dict(data=[])
        
        if 'orig_lmses' in kwargs.keys():
            if not bool(kwargs['orig_lmses']):
                raise ValueError("Original Landmarks cannot be None!")
            else:
                orig_lmses = kwargs['orig_lmses']
                assert len(orig_lmses) == len(img_paths), "The length of images and landmarks is not compatible!"
                
        if 'aligned_lmses' in kwargs.keys():
            if not bool(kwargs['aligned_lmses']):
                raise ValueError("Aligned Landmarks cannot be None!")
            else:
                aligned_lmses = kwargs['aligned_lmses']
                assert len(aligned_lmses) == len(img_paths), "The length of images and aligned landmarks is not compatible!"
        
        for i, (p, f) in enumerate(zip(img_paths, file_names)):
            fake_type = p.split('/')[-2] if self.fake_types != ['original'] else self.fake_types[0]
            img_obj = self._img_obj(p, f, id=i, fake_type=fake_type)
            
            if 'orig_lmses' in kwargs.keys(): 
                img_obj['orig_lms'] = orig_lmses[i].tolist() if isinstance(orig_lmses[i], np.ndarray) else orig_lmses[i] #To save to JSON
            if 'aligned_lmses' in kwargs.keys():
                img_obj['aligned_lms'] = aligned_lmses[i].tolist() if isinstance(aligned_lmses[i], np.ndarray) else aligned_lmses[i] #To save to JSON
            data["data"].append(img_obj)
        return data

    def save2json(self, data, fn='faceforensics_processed.json'):
        assert len(data), "Data can not be empty!"
        target = "processed_data/{}".format(self.compression)
        if not os.path.exists(target):
            os.mkdir(target)
        fp = os.path.join(target, fn)
        with open(fp, 'w') as f:
            json.dump(data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Landmarks preprocessing!')
    parser.add_argument('--config', help='Config file to proceed preprocessing')
    parser.add_argument('--file_path', help='File to load processed data')
    parser.add_argument('--extract_landmark', help='Use Dlib to extract landmarks', action='store_true')
    parser.add_argument('--save_aligned', help='Save aligned images', action='store_true')
    args = parser.parse_args()
    print(args)

    cfg = load_config(args.config)
    extract_landmark = args.extract_landmark
    save_aligned = args.save_aligned
    
    kwargs = {}
    if extract_landmark:
        kwargs['extract_landmark'] = extract_landmark

    #Initialize Landmark Utility instance
    if args.file_path:
        lm_ins = LandmarkUtility(cfg.PREPROCESSING, load_imgs=False, file_path=args.file_path, **kwargs)
    else:
        lm_ins = LandmarkUtility(cfg.PREPROCESSING, load_imgs=True, **kwargs)
    img_paths, file_names = lm_ins._load_data()
    print(f'{len(img_paths)} images have been loaded for processing!')
    
    if extract_landmark:
        assert cfg.PREPROCESSING.facial_lm_pretrained is not None, "Landmark pretrained can not be None!"
        f_detector = dlib.get_frontal_face_detector()
        f_lm_detector = dlib.shape_predictor(cfg.PREPROCESSING.facial_lm_pretrained)
        rot_imgs, f_lmses, rot_f_lmses = lm_ins.facial_landmarks(img_paths, f_detector, f_lm_detector)
        
        if save_aligned:
            os.makedirs(f'{lm_ins.image_root}{lm_ins.split}/{lm_ins.data_type}/aligned_{lm_ins.fake_types[0]}_{cfg.PREPROCESSING.N_LANDMARKS}', exist_ok=True)
            for i, img_p in enumerate(tqdm(img_paths, dynamic_ncols=True)):
                rot_img = rot_imgs[i]
                fn = file_names[i]
                vid_id = img_p.split('/')[-2]
                os.makedirs(f'{lm_ins.image_root}{lm_ins.split}/{lm_ins.data_type}/aligned_{lm_ins.fake_types[0]}_{cfg.PREPROCESSING.N_LANDMARKS}/{vid_id}', exist_ok=True)

                aligned_img_p = img_p.replace(lm_ins.fake_types[0], f'aligned_{lm_ins.fake_types[0]}_{cfg.PREPROCESSING.N_LANDMARKS}')
                cv2.imwrite(os.path.join(lm_ins.image_root, aligned_img_p), rot_img)
                img_paths[i] = aligned_img_p
        
        print('All landmarks have been detected and stored in memory!')
        print('Ready to save to file...')

    if args.file_path is None:
        if extract_landmark:
            data = lm_ins.build_data(img_paths, file_names, orig_lmses=f_lmses, aligned_lmses=rot_f_lmses)
        else:
            data = lm_ins.build_data(img_paths, file_names)
        
        try:
            lm_ins.save2json(data, fn=f'{lm_ins.split}_{lm_ins.dataset}_{lm_ins.fake_types[0]}_{cfg.PREPROCESSING.N_LANDMARKS}.json')
        except Exception as e:
            print(e)
        print("Processed Data has been saved successfully!")
