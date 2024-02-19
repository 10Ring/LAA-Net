#-*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch
import torch.nn as nn


layers_position = {
    'PoseResNet_50': 158,
    'PoseResNet_101': 311,
    'PoseEfficientNet_B4': 415,
}


def preset_model(cfg, model, optimizer=None):
    #Loading models from config, make sure the pretrained path correct to the model name
    start_epoch = 0
    if 'pretrained' in cfg.TRAIN and os.path.isfile(cfg.TRAIN.pretrained):
        model, optimizer, start_epoch = load_model(model, 
                                                   cfg.TRAIN.pretrained, 
                                                   optimizer=optimizer, 
                                                   resume=cfg.TRAIN.resume,
                                                   lr=cfg.TRAIN.lr,
                                                   lr_step=cfg.TRAIN.lr_scheduler.milestones,
                                                   gamma=cfg.TRAIN.lr_scheduler.gamma)
    else:
        model.init_weights(**cfg.MODEL.INIT_WEIGHTS)
    print('Loading model successfully -- {}'.format(cfg.MODEL.type))
    
    #Freeze backbone if begin_epoch < warm up
    if cfg.TRAIN.freeze_backbone and start_epoch < cfg.TRAIN.warm_up:
        freeze_backbone(cfg.MODEL, model)
    
    print('Number of parameters', sum(p.numel() for p in model.parameters()))
    print('Number of trainable parameters', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model, optimizer, start_epoch


def load_pretrained(model, weight_path):
    '''
    This function only care about state dict of model
    For other modules such as optimizer, resume learning, please refer @load_model
    '''
    state_dict = torch.load(weight_path)['state_dict']
    model.load_state_dict(state_dict, strict=True)
    return model


def freeze_backbone(cfg, model):
    '''
    This func to freeze some specific layers to warm up the models
    '''
    if hasattr(model, 'backbone'):
        backbone = model.backbone
        for param in backbone.parameters():
            param.requires_grad = False
    else:
        for i, (n, p) in enumerate(model.named_parameters()):
            if (i <= layers_position[f'{cfg.type}_{cfg.num_layers}']):
                p.requires_grad = False


def unfreeze_backbone(model):
    '''
    This func to unfreeze all model layers
    '''
    for param in model.parameters():
        if not param.requires_grad:
            param.requires_grad = True


def load_model(model, model_path, optimizer=None, resume=False, 
               lr=None, lr_step=None, gamma=None):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}
  
    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    msg = 'If you see this, your model does not fully load the ' + \
            'pre-trained weight. Please make sure ' + \
            'you have correctly specified --arch xxx ' + \
            'or set the correct --num_classes for your own dataset.'
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, '\
                      'loaded shape{}. {}'.format(
                        k, model_state_dict[k].shape, state_dict[k].shape, msg))
                state_dict[k] = model_state_dict[k]
            else:
                print('Drop parameter {}.'.format(k) + msg)
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k) + msg)
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            start_lr = lr
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')
    return model, optimizer, start_epoch


def save_model(path, epoch, model, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch,
            'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)
