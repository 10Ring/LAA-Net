#-*- coding: utf-8 -*-
from __future__ import absolute_import
import time

import os
import sys
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
import argparse
from datetime import datetime
import random

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import numpy as np

from configs.get_config import load_config
from models import *
from datasets import *
from losses import *
from lib.core_function import validate, train, test
from logs.logger import Logger, LOG_DIR
from lib.optimizers.sam import SAM
from lib.scheduler.linear_decay import LinearDecayLR


def args_parser(args=None):
    parser = argparse.ArgumentParser("Training process...")
    parser.add_argument('--cfg', help='Config file', required=True)
    parser.add_argument('--alloc_mem', '-a',  help='Pre allocating GPU memory', action='store_true')
    return parser.parse_args(args)


if __name__=='__main__':
    if len(sys.argv[1:]):
        args = sys.argv[1:]
    else:
        args = None
    
    args = args_parser(args)
    cfg = load_config(args.cfg)
    logger = Logger(task=f'training_{cfg.TASK}')

    #Seed
    seed = cfg.SEED
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    # Allocate memory
    if args.alloc_mem:
        mem_all_tensors = torch.rand(60,10000,10000)
        mem_all_tensors.to('cuda:0')

    #Configuing GPU devices
    devices = torch.device('cpu')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    if 'gpus' in cfg.TRAIN.gpus and cfg.TRAIN.gpus is not None:
        #Only support a single gpu for training now
        devices = torch.device('cuda:1')
    model = build_model(cfg.MODEL, MODELS).cuda().to(torch.float64)
    
    #Loading Dataloader
    start_loading = time.time()
    val_dataset = build_dataset(cfg.DATASET, 
                                DATASETS, 
                                default_args=dict(split='val', config=cfg.DATASET))
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=cfg.TRAIN.batch_size * len(cfg.TRAIN.gpus),
                                shuffle=False,
                                pin_memory=cfg.DATASET.PIN_MEMORY,
                                num_workers=cfg.DATASET.NUM_WORKERS,
                                worker_init_fn=val_dataset.train_worker_init_fn,
                                collate_fn=val_dataset.train_collate_fn)
                                
    logger.info('Loading val dataloader successfully! -- {}'.format(time.time() - start_loading))
    
    start_loading = time.time()
    train_dataset = build_dataset(cfg.DATASET,
                                  DATASETS,
                                  default_args=dict(split='train', config=cfg.DATASET))
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=cfg.TRAIN.batch_size * len(cfg.TRAIN.gpus),
                                  shuffle=True,
                                  pin_memory=cfg.DATASET.PIN_MEMORY,
                                  num_workers=cfg.DATASET.NUM_WORKERS,
                                  worker_init_fn=train_dataset.train_worker_init_fn,
                                  collate_fn=train_dataset.train_collate_fn)
    logger.info('Loading Train dataloader successfully! -- {}'.format(time.time() - start_loading))
    
    start_loading = time.time()
    test_dataset = build_dataset(cfg.DATASET, 
                                 DATASETS,
                                 default_args=dict(split='test', config=cfg.DATASET))
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=cfg.TRAIN.batch_size * len(cfg.TRAIN.gpus),
                                 shuffle=True,
                                 pin_memory=cfg.DATASET.PIN_MEMORY,
                                 num_workers=cfg.DATASET.NUM_WORKERS)
    logger.info('Loading Test dataloader successfully! -- {}'.format(time.time() - start_loading))
    
    #Defining Loss function and Optimizer
    critetion = build_losses(cfg.TRAIN.loss, LOSSES, default_args=dict(cfg=cfg.TRAIN.loss)).cuda().to(torch.float64)
    if cfg.TRAIN.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.lr, weight_decay=2e-5)
    elif cfg.TRAIN.optimizer == 'SAM':
        optimizer = SAM(model.parameters(), optim.Adam, lr=cfg.TRAIN.lr, weight_decay=1e-4)
    else:
        optimizer = optim.SGD(model.parameters(), lr=cfg.TRAIN.lr, weight_decay=1e-5, momentum=0.9)
        
    #Loading model
    model, optimizer, start_epoch = preset_model(cfg, model, optimizer=optimizer)
    if len(cfg.TRAIN.gpus) > 0:
        model = nn.DataParallel(model, device_ids=cfg.TRAIN.gpus).cuda().to(torch.float64)
    else:
        model = model.cuda().to(torch.float64)
    
    #Learning rate Scheduler
    if cfg.TRAIN.lr_scheduler == 'MultiStepLR':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **cfg.TRAIN.lr_scheduler)
    else:
        lr_scheduler = LinearDecayLR(optimizer, cfg.TRAIN.epochs, cfg.TRAIN.epochs//4, last_epoch=cfg.TRAIN.begin_epoch, booster=4)

    #Enabling tensorboard
    writer = SummaryWriter('.tensorboard/{}_{}'.format(datetime.today().strftime('%Y-%m-%d'), cfg.TASK))
    
    trainIters = 0
    valIters = 0
    min_val_loss = 1e10
    max_val_acc = 0
    max_test_auc = 0
    metrics_base = cfg.METRICS_BASE # Combine heatmap + cls prediction to calculate accuracy
    
    #Starting training process
    logger.info('Starting training process...')
    for epoch in range(start_epoch, cfg.TRAIN.epochs):
        #Unfreezin backbone to update weights
        if cfg.TRAIN.freeze_backbone and epoch == cfg.TRAIN.warm_up:
            unfreeze_backbone(model)
        
        np.random.seed(seed + epoch)
        if epoch > 0 and cfg.DATA_RELOAD:
            logger.info(f'Reloading data for epoch {epoch}...')
            train_dataset._reload_data()
            train_dataloader = DataLoader(train_dataset, 
                                          batch_size=cfg.TRAIN.batch_size * len(cfg.TRAIN.gpus),
                                          shuffle=True,
                                          pin_memory=cfg.DATASET.PIN_MEMORY,
                                          num_workers=cfg.DATASET.NUM_WORKERS,
                                          worker_init_fn=train_dataset.train_worker_init_fn,
                                          collate_fn=train_dataset.train_collate_fn)
            
        loss_avg, acc_avg, trainIters = train(cfg, 
                                              model, 
                                              critetion, 
                                              optimizer, 
                                              epoch, 
                                              train_dataloader, 
                                              logger, 
                                              writer, 
                                              devices, 
                                              trainIters, 
                                              metrics_base=metrics_base)
        if epoch % cfg.TRAIN.every_val_epochs == 0:
            loss_val, acc_val, valIters = validate(cfg, 
                                                   model, 
                                                   critetion, 
                                                   epoch, 
                                                   val_dataloader, 
                                                   logger, 
                                                   writer, 
                                                   devices, 
                                                   valIters, 
                                                   metrics_base=metrics_base)

            if ((acc_val.avg > max_val_acc) and (loss_val.avg < min_val_loss)):
                # Saving checkpoint 
                ckp_path = os.path.join(LOG_DIR, '{}_{}_model_best.pth'.format(cfg.MODEL.type, cfg.TASK))
                save_model(path=ckp_path, epoch=epoch, model=model, optimizer=optimizer)
                min_val_loss = loss_val.avg
                max_val_acc = acc_val.avg
                logger.info(f'Saved best model at epoch --- {epoch}')
        lr_scheduler.step()
