#-*- coding: utf-8 -*-
import os
import random
import sys
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

import time
import argparse
import multiprocessing as mp
from threading import Thread
import queue

from skimage import io
from skimage import transform as sktransform
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from imgaug import augmenters as iaa
import cv2
from tqdm import tqdm

from package_utils.deepfake_mask import random_get_hull
from package_utils.utils import save_file, load_file
# from datasets.sbi.utils import gen_target

IMAGE_H, IMAGE_W, IMAGE_CHANNEL = 256, 256, 3
COMPRESSION = 'c0'
SPLIT = 'train'
DATA_TYPE = 'frames'
IMAGE_ROOT = f'/data/deepfake_cluster/datasets_df/FaceForensics++/{COMPRESSION}/'
ANNO_FILE = 'processed_data/train_faceforensics_processed.json'
DEST_DIR = 'FaceXRay'
LABEL_FILE = 'train_FF_FaceXRay.json'
NUMBER_OF_PROCESS = 4
BLENDING_TYPE = 'BI'
MARGIN = 20


def args_parse(args=None):
    args_parser = argparse.ArgumentParser('Blending Image Processing Hub...')
    args_parser.add_argument('--task', '-t', help="Defining task!")
    args_parser.add_argument('--anno_file', '-f', help="Pre annotation file!")
    args_parser.add_argument('--fake_type', '-ft', help="Faketype to manipulation!")
    args_parser.add_argument('--mp', '-m', help="Apply multiprocessing", action="store_true")
    args = args_parser.parse_args(args)
    return args


def name_resolve(path):
    if COMPRESSION == 'c23':
        name = os.path.splitext(os.path.basename(path))[0]
        vid_id, frame_id = name.split('_')[0:2]
    else:
        name = path.split('/')
        vid_id, frame_id = name[-2], os.path.splitext(name[-1])[0]
    return vid_id, frame_id


def gen_real_fimg_mask(img_path):
    face_img = io.imread(img_path)
    mask = np.zeros((face_img.shape[0], face_img.shape[1], 3))
    mask = (mask*255).astype(np.uint8)
    return face_img, mask

    
def total_euclidean_distance(a,b):
    assert len(a.shape) == 2
    return np.sum(np.linalg.norm(a-b,axis=1))


def random_erode_dilate(mask, ksize=None):
    if random.random()>0.5:
        if ksize is  None:
            ksize = random.randint(1,21)
        if ksize % 2 == 0:
            ksize += 1
        mask = np.array(mask).astype(np.uint8)*255
        kernel = np.ones((ksize,ksize),np.uint8)
        mask = cv2.erode(mask, kernel, 1)/255
    else:
        if ksize is  None:
            ksize = random.randint(1,5)
        if ksize % 2 == 0:
            ksize += 1
        mask = np.array(mask).astype(np.uint8)*255
        kernel = np.ones((ksize, ksize),np.uint8)
        mask = cv2.dilate(mask, kernel, 1)/255
    return mask


# borrow from https://github.com/MarekKowalski/FaceSwap
def blendImages(src, dst, mask, featherAmount=0.2):
   
    maskIndices = np.where(mask != 0)
    
    src_mask = np.ones_like(mask)
    dst_mask = np.zeros_like(mask)

    maskPts = np.hstack((maskIndices[1][:, np.newaxis], maskIndices[0][:, np.newaxis]))
    faceSize = np.max(maskPts, axis=0) - np.min(maskPts, axis=0)
    featherAmount = featherAmount * np.max(faceSize)

    hull = cv2.convexHull(maskPts)
    dists = np.zeros(maskPts.shape[0])
    
    for i in range(maskPts.shape[0]):
        dists[i] = cv2.pointPolygonTest(hull, (maskPts[i, 0], maskPts[i, 1]), True)

    weights = np.clip(dists / featherAmount, 0, 1)

    composedImg = np.copy(dst)
    composedImg[maskIndices[0], maskIndices[1]] = weights[:, np.newaxis] * src[maskIndices[0], maskIndices[1]] + (1 - weights[:, np.newaxis]) * dst[maskIndices[0], maskIndices[1]]

    composedMask = np.copy(dst_mask)
    composedMask[maskIndices[0], maskIndices[1]] = weights[:, np.newaxis] * src_mask[maskIndices[0], maskIndices[1]] + (
                1 - weights[:, np.newaxis]) * dst_mask[maskIndices[0], maskIndices[1]]

    return composedImg, composedMask


# borrow from https://github.com/MarekKowalski/FaceSwap
def colorTransfer(src, dst, mask):
    transferredDst = np.copy(dst)
    
    maskIndices = np.where(mask != 0)
    

    maskedSrc = src[maskIndices[0], maskIndices[1]].astype(np.int32)
    maskedDst = dst[maskIndices[0], maskIndices[1]].astype(np.int32)

    meanSrc = np.mean(maskedSrc, axis=0)
    meanDst = np.mean(maskedDst, axis=0)

    maskedDst = maskedDst - meanDst
    maskedDst = maskedDst + meanSrc
    maskedDst = np.clip(maskedDst, 0, 255)

    transferredDst[maskIndices[0], maskIndices[1]] = maskedDst

    return transferredDst


class BIOnlineGeneration():
    def __init__(self, data_record, queue_size=1024, mlprocess=False, number=1, fake_type=None):
        self.landmarks_record = {}
        self.data_record = data_record['data']
        self.mlprocess = mlprocess
        self.number = number
        self.fake_type = fake_type
        
        if self.fake_type is not None:
            self.data_record = self._filter_data()
            if not len(self.data_record):
                raise ValueError('DataList can not be Empty!')
        
        print(f'You are generating data for --- {self.fake_type} --- {len(self.data_record)} images')
        
        for item in self.data_record:
            if 'aligned_lms' in item.keys() and len(item['aligned_lms']):
                self.landmarks_record[item['image_path']] = np.array(item['aligned_lms'])
            else:
                self.landmarks_record[item['image_path']] = np.array(item['orig_lms'])
        
        # extract all frame from all video in the name of {videoid}_{frameid}
        self.data_list = [item['image_path'] for item in self.data_record]
        self.file_names = [item['file_name'] for item in self.data_record]
        
        if COMPRESSION != 'c23':
            self.labels = [item['image_path'].split('/')[-3] for item in self.data_record]
        else:
            self.labels = [item['image_path'].split('/')[-2] for item in self.data_record]
        self.vid_ids = [item['image_path'].split('/')[-2] for item in self.data_record]
        
        # predefine mask distortion
        self.distortion = iaa.Sequential([iaa.PiecewiseAffine(scale=(0.01, 0.15))])
        
        self.result_queue = mp.Queue(maxsize=queue_size)
        self.final_results = []

    def register_task(self, target):
        if self.mlprocess:
            p = mp.Process(target=target, args=())
        else:
            p = Thread(target=target, args=())
        return p
        
    def start(self, p):
        self.result_worker = p
        self.result_worker.start()
        
    def wait_n_put(self, item):
        self.result_queue.put(item)
        
    def wait_n_get(self):
        return self.result_queue.get()
        
    def count(self):
        return self.result_queue.qsize()
    
    def stop(self):
        self.result_worker.join()
        
    def running(self):
        return not self.result_queue.empty()

    def terminate(self):
        self.result_worker.terminate()
        
    def clear(self):
        while not self.result_queue.empty():
            self.result_queue.get()
    
    def clear_sequences(self):
        self.clear()
        
    def get_results(self):
        all_objs = []
        
        while not self.final_results.empty():
            all_objs.append(self.final_results.get())
        return all_objs
    
    def _filter_data(self):
        assert self.fake_type is not None, "Fake type is require to filter data!"
        assert self.fake_type in ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
        
        self.data_record = [item for item in self.data_record if item['fake_type'] == self.fake_type]
        return self.data_record
        
    def gen_one_datapoint(self, idx):
        background_face_path = self.data_list[idx]
        label = self.labels[idx]
        
        # Choose Blending type
        if BLENDING_TYPE != 'SBI':
            data_type = 'real' if random.randint(0,1) else 'fake'
        else:
            data_type = 'fake'
        
        # Handle for the cases of blank landmarks, auto real if real label is real image, otherwise return
        if not self.landmarks_record[background_face_path].any():
            if not ('fake' == label):
                data_type = 'real'
            else:
                return None, None, None
            
        if data_type == 'fake':
            if BLENDING_TYPE != 'SBI':
                face_img, mask =  self.get_blended_face(background_face_path, self.landmarks_record[background_face_path])
            else:
                face_img, mask, face_r, mask_r = gen_target(os.path.join(IMAGE_ROOT, background_face_path), self.landmarks_record[background_face_path], margin=[MARGIN, MARGIN])
        else:
            face_img, mask = gen_real_fimg_mask(os.path.join(IMAGE_ROOT, background_face_path))

        face_img = face_img[MARGIN:IMAGE_H-MARGIN, MARGIN:IMAGE_W-MARGIN, :]
        mask = mask[MARGIN:IMAGE_H-MARGIN, MARGIN:IMAGE_W-MARGIN, :]

        return face_img, mask, data_type
        
    def get_blended_face(self, background_face_path, background_landmark):
        background_face = io.imread(os.path.join(IMAGE_ROOT, background_face_path))
        
        foreground_face_path = self.search_similar_face(background_landmark, background_face_path, get_best=True)
        foreground_face = io.imread(os.path.join(IMAGE_ROOT, foreground_face_path))
        
        # down sample before blending
        img_h, img_w = background_face.shape[:2]
        aug_size = random.randint(img_h//2, img_h)
        background_landmark = background_landmark * (aug_size/img_h)
        
        foreground_face = sktransform.resize(foreground_face,(aug_size,aug_size),preserve_range=True).astype(np.uint8)
        background_face = sktransform.resize(background_face,(aug_size,aug_size),preserve_range=True).astype(np.uint8)
        
        # get random type of initial blending mask
        mask = random_get_hull(background_landmark, background_face)
       
        #  random deform mask
        mask = self.distortion.augment_image(mask)
        mask = random_erode_dilate(mask)
        
        # filte empty mask after deformation
        if np.sum(mask) == 0 :
            print(f'There was an issue when doing blending with Image -- {background_face_path}')
            print(f'Reverting by returning a real image and mask...')
            face_img, mask = gen_real_fimg_mask(os.path.join(IMAGE_ROOT, background_face_path))
            return face_img, mask

        # apply color transfer
        foreground_face = colorTransfer(background_face, foreground_face, mask*255)
        
        # blend two face
        blended_face, mask = blendImages(foreground_face, background_face, mask*255)
        blended_face = blended_face.astype(np.uint8)
       
        # resize back to default resolution
        blended_face = sktransform.resize(blended_face,(img_h, img_w),preserve_range=True).astype(np.uint8)
        mask = sktransform.resize(mask,(img_h, img_w),preserve_range=True)
        mask = mask[:,:,0:1]
        mask = (1 - mask) * mask * 4
        mask = np.repeat(mask,3,2)
        mask = (mask*255).astype(np.uint8)
        
        # randomly downsample after BI pipeline
        face_img = Image.fromarray(blended_face)
        if random.randint(0,1):
            aug_size = random.randint(img_h//4, img_h)
            if random.randint(0,1):
                face_img = face_img.resize((aug_size, aug_size), Image.BILINEAR)
            else:
                face_img = face_img.resize((aug_size, aug_size), Image.NEAREST)
            face_img = face_img.resize((img_h, img_w),Image.BILINEAR)
        face_img = np.array(face_img)
            
        # # random jpeg compression after BI pipeline
        # if random.randint(0,1):
        #     quality = random.randint(60, 100)
        #     encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        #     face_img_encode = cv2.imencode('.jpg', face_img, encode_param)[1]
        #     face_img = cv2.imdecode(face_img_encode, cv2.IMREAD_COLOR)
        
        # # random flip
        # if random.randint(0,1):
        #     face_img = np.flip(face_img,1)
        #     mask = np.flip(mask,1)
            
        return face_img, mask
    
    def search_similar_face(self, this_landmark, background_face_path, get_best=False):
        vid_id, frame_id = name_resolve(background_face_path)
        min_dist = 99999999
        
        # random sample 5000 frame from all frams:
        all_candidate_path = random.sample(self.data_list, k=10000)
        
        # filter all frame that comes from the same video as background face
        all_candidate_path = filter(lambda k:name_resolve(k)[0] != vid_id, all_candidate_path)
        all_candidate_path = list(all_candidate_path)
        candidate_distance_list = {}
        
        # loop throungh all candidates frame to get best match
        for candidate_path in all_candidate_path:
            candidate_landmark = self.landmarks_record[candidate_path].astype(np.float32)
            
            if not candidate_landmark.any():
                continue
            
            candidate_distance = total_euclidean_distance(candidate_landmark, this_landmark)
            if candidate_distance < min_dist:
                min_dist = candidate_distance
                min_path = candidate_path
            candidate_distance_list[candidate_path] = candidate_distance
        if not get_best:
            return candidate_distance_list
        else:
            return min_path
    
    def search_similar_faces(self):
        while True:
            item = self.wait_n_get()
            if item is None:
                obj_list = self.final_results
                data = {"data": obj_list}
                save_file(data, f'processed_data/{COMPRESSION}/dynamic_trainBI_FFv4.json')
                return True
            bg_path = item['image_path']
            if 'aligned_lms' in item.keys() and len(item['aligned_lms']):
                f_lms = np.array(item['aligned_lms'])
            else:
                f_lms = np.array(item['orig_lms'])
            best_match_paths = []
            
            if f_lms.any():
                candidate_list = self.search_similar_face(f_lms, bg_path)
                best_match_paths = sorted(candidate_list.items(), key=lambda x: x[1], reverse=False)[:self.number]
                best_match_paths = [it[0] for it in best_match_paths]
            item["best_match"] = best_match_paths
            self.final_results.append(item)


if __name__ == '__main__':
    if sys.argv[1:] is not None:
        args = args_parse(sys.argv[1:])
    else:
        args = sys.argv[:-1]
    
    task = args.task
    anno_file = args.anno_file
    mp_ = args.mp
    fake_type = args.fake_type
    assert len(anno_file), "Annotation file path can not be empty!"
    assert os.access(anno_file, os.R_OK), "Annotation file path must be valid to access!"

    print("Starting to load processed data...")
    start = time.time()
    data_record = load_file(anno_file)
    print('Loading time --- {}'.format(time.time() - start))
    ds = BIOnlineGeneration(data_record, mlprocess=mp_, number=30, fake_type=fake_type)
    data = {}

    assert task in ['save_blending', 'search_similar_lms']
    if task == 'save_blending':
        all_object = []
        
        for i in tqdm(range(len(ds.data_list))):
            img, mask, label = ds.gen_one_datapoint(i)
            if img is None and mask is None and label is None:
                continue
            
            if COMPRESSION == 'c23':
                image_path = os.path.join(IMAGE_ROOT, DEST_DIR, 'images', ds.file_names[i])
                mask_path = os.path.join(IMAGE_ROOT, DEST_DIR, 'masks', ds.file_names[i])
            else:
                image_path = os.path.join(IMAGE_ROOT, SPLIT, DATA_TYPE, DEST_DIR, 'images', f'{ds.vid_ids[i]}_{ds.file_names[i]}')
                mask_path = os.path.join(IMAGE_ROOT, SPLIT, DATA_TYPE, DEST_DIR, 'masks', f'{ds.vid_ids[i]}_{ds.file_names[i]}')
            
            try:
                mask_pil = Image.fromarray(mask)
                mask_pil.save(mask_path)
                
                image = Image.fromarray(img)
                image.save(image_path)
            except Exception as e:
                print(e)
                continue
            
            all_object.append({
                'id': i,
                'image_path': image_path,
                'mask_path': mask_path,
                'label': label
            })
        data["data"] = all_object
        save_file(data, file_path=os.path.join(IMAGE_ROOT, SPLIT, DATA_TYPE, DEST_DIR, LABEL_FILE))
    elif task == 'search_similar_lms':
        p = ds.register_task(ds.search_similar_faces)
        ds.start(p)
        
        try:
            for i, item in enumerate(tqdm(ds.data_record, dynamic_ncols=True)):
                ds.wait_n_put(item)
            ds.wait_n_put(None)
            
            while (ds.running()):
                time.sleep(1)
                print('===============> Rendering remaining ' + str(ds.count()) + ' images in the queue...', end='\r')
            ds.stop()
        except Exception as e:
            print(repr(e))
            print("There is an exception during process! Please check it")
        except KeyboardInterrupt:
            ds.terminate()
            ds.clear_sequences()
        exit(0)
    else:
        raise ValueError('This task {} is not supported at the moment!')
