#-*- coding: utf-8 -*-
from glob import glob
import os
import pandas as pd
import cv2
from tqdm import tqdm
import numpy as np
import shutil
import json
import sys
import argparse
from imutils import face_utils
from retinaface.pre_trained_models import get_model
from retinaface.utils import vis_annotations
import torch


ROOT = '/data/deepfake_cluster/datasets_df'
SAVE_DIR = f'{ROOT}/FaceForensics++/c0'
IMAGE_H, IMAGE_W, IMAGE_C = 256, 256, 3
PADDING = 0.25


def facecrop(model, org_path, save_path, period=1, num_frames=10, dataset='original', label=None, mask_path=None, padding=PADDING):
    print(f'Processing video --- {org_path}')
    cap_org = cv2.VideoCapture(org_path)
    if mask_path is not None:
        mask_cap = cv2.VideoCapture(mask_path)
    croppedfaces=[]
    frame_count_org = int(cap_org.get(cv2.CAP_PROP_FRAME_COUNT))
    print("N frame count --- ", frame_count_org)
    
    if label is not None:
        save_path_ = save_path + f'/frames/{dataset}/{str(label)}/' + os.path.basename(org_path).replace('.mp4','/')
    else:
        save_path_ = save_path + f'/frames/{dataset}/' + os.path.basename(org_path).replace('.mp4','/')
    os.makedirs(save_path_, exist_ok=True)

    if mask_path is not None:
        save_mask_path_ = save_path + f'/masks/{dataset}/' + os.path.basename(mask_path).replace('.mp4','/')
        os.makedirs(save_mask_path_, exist_ok=True)
    
    frame_idxs = np.linspace(0, frame_count_org - 1, num_frames, endpoint=True, dtype=np.int64)
    
    for cnt_frame in range(frame_count_org):
        image_path = save_path_+str(cnt_frame).zfill(3)+'.png'
        if os.path.isfile(image_path): continue
        if mask_path is not None:
            mask_f_path = save_mask_path_+str(cnt_frame).zfill(3)+'.png'
            if os.path.isfile(mask_f_path): continue
        
        try:
            ret_org, frame_org = cap_org.read()
            if mask_path is not None:
                ret_m_org, mask_org = mask_cap.read()
            height, width = frame_org.shape[:-1]
            if not ret_org:
                tqdm.write('Frame read {} Error! : {}'.format(cnt_frame,os.path.basename(org_path)))
                continue
            
            if cnt_frame not in frame_idxs:
                continue
            
            frame = cv2.cvtColor(frame_org, cv2.COLOR_BGR2RGB)
            faces = model.predict_jsons(frame)
            try:
                if len(faces)==0:
                    print(faces)
                    tqdm.write('No faces in {}:{}'.format(cnt_frame,os.path.basename(org_path)))
                    continue
                
                face_s_max = -1
                landmarks = []
                face_crop = None
                score_max = -1
                for face_idx in range(len(faces)):
                    x0,y0,x1,y1 = faces[face_idx]['bbox']
                    # landmark = np.array([[x0,y0],[x1,y1]] + faces[face_idx]['landmarks'])
                    face_w = x1 - x0
                    face_h = y1 - y0
                    face_s = face_w * face_h
                    score = faces[face_idx]['score']                    
                    
                    if face_s > face_s_max and score > score_max:
                        f_c_x0 = max(0, x0 - int(face_w*padding))
                        f_c_x1 = min(width, x1 + int(face_w*padding))
                        f_c_y0 = max(0, y0 - int(face_h*padding))
                        f_c_y1 = min(height, y1 + int(face_h*padding))
                        
                        face_crop = frame_org[f_c_y0:f_c_y1, f_c_x0:f_c_x1, :]
                        if mask_path is not None:
                            mask_crop = mask_org[f_c_y0:f_c_y1, f_c_x0:f_c_x1, :]
                        face_s_max = face_s
                        score_max = score
                        # size_list.append(face_s)
                        # # landmarks.append(landmark)
            except Exception as e:
                print(f'error in {cnt_frame}:{org_path}')
                print(e)
                continue
        except Exception as e1:
            print(e1)
            continue
            
        # landmarks=np.concatenate(landmarks).reshape((len(size_list),) + landmark.shape)
        # landmarks=landmarks[np.argsort(np.array(size_list))[::-1]]
        
        # land_path=save_path_+str(cnt_frame).zfill(3)
        # land_path=land_path.replace('/frames','/retina')
        # os.makedirs(os.path.dirname(land_path),exist_ok=True)
        # np.save(land_path, landmarks)
        # if not os.path.isfile(image_path):
        face_crop = cv2.resize(face_crop, (IMAGE_H, IMAGE_W), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(image_path, face_crop)
        
        if mask_path is not None:
            mask_crop = cv2.resize(mask_crop, (IMAGE_H, IMAGE_W), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(mask_f_path, mask_crop)
        
    cap_org.release()
    return


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-d', dest='dataset', 
                              choices=['FaceShifter','Face2Face','Deepfakes','FaceSwap','NeuralTextures',\
                                       'Original','Celeb-real','Celeb-synthesis','YouTube-real','DFDC','DFDCP','method_A','method_B','original_videos'])
    parser.add_argument('-c', dest='comp', choices=['raw','c23','c40'], default='raw')
    parser.add_argument('-n', dest='num_frames', type=int,default=32)
    parser.add_argument('-t', dest='task', choices=['train', 'val', 'test'], default='train')
    parser.add_argument('--save_mask', '-sm', action='store_true')
    parser.add_argument('--alloc_mem', '-a',  help='Pre allocating GPU memory', action='store_true')
    args=parser.parse_args()

    # Allocate memory
    if args.alloc_mem:
        mem_all_tensors = torch.rand(60,10000,10000)
        mem_all_tensors.to('cuda:0')

    # Setting device
    device=torch.device('cuda')

    # Setting the dataset path based on the dataset name
    if args.dataset=='Original':
        dataset_path='{}/FaceForensics++/original_download/original_sequences/youtube/'.format(ROOT)
    elif args.dataset=='DeepFakeDetection_original':
        dataset_path='/data/FaceForensics++/original_sequences/actors/{}/'.format(args.comp)
    elif args.dataset in ['DeepFakeDetection','FaceShifter','Face2Face','Deepfakes','FaceSwap','NeuralTextures']:
        dataset_path='{}/FaceForensics++/original_download/manipulated_sequences/{}/'.format(ROOT, args.dataset)
    elif args.dataset in ['Celeb-real','Celeb-synthesis','YouTube-real']:
        if 'v1' in SAVE_DIR:
            dataset_path='{}/Celeb-DFv1/'.format(ROOT)
        else:
            dataset_path='{}/Celeb-DFv2/Celeb-DF-v2/'.format(ROOT)
    elif args.dataset in ['method_A','method_B','original_videos']:
        dataset_path='{}/DFDCP/'.format(ROOT)
    elif args.dataset in ['DFDC']:
        dataset_path='{}/DFDC/'.format(ROOT)
    else:
        raise NotImplementedError
    
    # Loading model
    model = get_model("resnet50_2020-07-20", max_size=2048, device=device)
    model.eval()

    labels = []
    if args.dataset in ['Original','DeepFakeDetection','FaceShifter','Face2Face','Deepfakes','FaceSwap','NeuralTextures']:
        movies_path = os.path.join(dataset_path, args.comp, 'videos/')
        mask_mov_paths = os.path.join(dataset_path, 'masks', 'videos/')
        
        # Annotation file for FF++
        with open(f'{ROOT}/FaceForensics++/original_download/{args.task}.json') as f:
            vid_ids = json.load(f)
    elif args.dataset in ['Celeb-real', 'Celeb-synthesis', 'YouTube-real']:
        if 'v1' in SAVE_DIR:
            movies_path = dataset_path
        else:
            movies_path = os.path.join(dataset_path, args.dataset, 'videos')
            
        # Annotation file for Celeb-DF
        with open(f'{dataset_path}List_of_{args.task}ing_videos.txt') as f:
            vid_ids = pd.read_csv(f).values.reshape(-1)
    elif args.dataset in ['DFDC']:
        movies_path = os.path.join(dataset_path, args.task, 'videos')
        
        # Annotation file for DFDC
        with open(os.path.join(dataset_path, args.task, 'labels.csv')) as f:
            df = pd.read_csv(f)
            # df['path'] = df['label'].astype(str) + '/' + df['filename']
            vid_ids = df['filename'].values.reshape(-1)
            labels = df['label'].values.reshape(-1) 
    else:
        movies_path = dataset_path
        
        # Annotation file for DFDCP
        with open(f'{ROOT}/DFDCP/dataset.json') as f:
            movie_data = json.load(f)
        vid_ids = []
        for mv_id, item_data in movie_data.items():
            if item_data["set"] == args.task:
                vid_ids.append(mv_id)
    
    movies_path_list = []
    mask_mov_path_list = []
    file_list = []
    
    # Loading the list of specific video's names for an invidual task 'train/val/test
    for i in range(len(vid_ids)):    
        if args.dataset == 'Original':   
            file_list += vid_ids[i]
        elif args.dataset in ['Face2Face','Deepfakes','FaceSwap','NeuralTextures']:
            file_list.append('_'.join([vid_ids[i][0], vid_ids[i][1]]))
            file_list.append('_'.join([vid_ids[i][1], vid_ids[i][0]]))
        elif args.dataset in ['method_A','method_B','original_videos']:
            if args.dataset in vid_ids[i]:
                file_list.append(vid_ids[i])
        elif args.dataset in ['DFDC']:
            file_list.append(vid_ids[i])
        else:
            if args.dataset in vid_ids[i]:
                file_list.append(vid_ids[i].split(' ')[-1])
    
    # movies_path_list = sorted(glob(movies_path+'*.mp4'))
    if args.dataset in ['Original','DeepFakeDetection','FaceShifter','Face2Face','Deepfakes','FaceSwap','NeuralTextures']:
        [movies_path_list.append(movies_path+i+'.mp4') for i in file_list]
        [mask_mov_path_list.append(mask_mov_paths+i+'.mp4') for i in file_list]
    else:
        if 'v2' in SAVE_DIR:
            [movies_path_list.append(os.path.join(movies_path, i.split('/')[-1])) for i in file_list]
        else:
            [movies_path_list.append(os.path.join(movies_path, i)) for i in file_list]
    
    print("{} : videos are exist in {}".format(len(movies_path_list), args.dataset))
    n_sample=len(movies_path_list)
    print(f'number of video samples -- {n_sample}')

    # Defining the path to store the images
    save_path = os.path.join(SAVE_DIR, args.task)
    os.makedirs(save_path, exist_ok=True)
    
    for i in tqdm(range(0, n_sample)):
        # folder_path=movies_path_list[i].replace('videos/','frames/').replace('.mp4','/')
        # if len(glob(folder_path.replace('/frames/','/retina/')+'*.npy')) < args.num_frames:
        if len(labels):
            facecrop(model, movies_path_list[i], save_path=save_path, num_frames=args.num_frames, dataset=args.dataset, label=labels[i])
        else:
            if not args.save_mask:
                facecrop(model, movies_path_list[i], save_path=save_path, num_frames=args.num_frames, dataset=args.dataset)
            else:
                facecrop(model, movies_path_list[i], save_path=save_path, num_frames=args.num_frames, dataset=args.dataset, mask_path=mask_mov_path_list[i])
