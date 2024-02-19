#-*- coding: utf-8 -*-
import os
import numpy as np
from sklearn import metrics
import torch
from sklearn.metrics import average_precision_score, recall_score

from losses.losses import _avg_sigmoid, _sigmoid


def bin_calculate_acc(preds, labels, targets=None, threshold=0.5):
    if preds.shape[-1] > 1:
        preds = preds.softmax(dim=-1)
        preds = preds[:, -1]
    else:
        preds = preds.sigmoid()
    
    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    preds_ = (preds >= threshold).astype(int)
    acc = np.mean((preds_ == labels).astype(int), axis=0)
    
    if acc.ndim >= 1:
        return acc[0]
    else:
        return acc


def hm_calculate_acc(preds, targets=None, labels=None, threshold=0.5):
    cls_ = _avg_sigmoid(preds)
    acc = bin_calculate_acc(cls_, labels, threshold=threshold)
    return acc


def hm_bin_calculate_acc(hm_preds, cls_preds, targets=None, labels=None, cls_lamda=0.05):
    # Select top hm_preds
    hm_preds_ = _sigmoid(hm_preds.clone())
    hm_preds_ = torch.reshape(hm_preds_, (hm_preds_.shape[0], hm_preds_.shape[1], -1))
    top_k = torch.topk(hm_preds_, 10, -1).values
    mean_hm_preds = torch.mean(top_k, -1)
    
    cls_preds_ = cls_lamda*cls_preds + (1-cls_lamda)*mean_hm_preds
    acc = bin_calculate_acc(cls_preds_, labels)
    return acc


def bin_calculate_auc_ap_ar(cls_preds, labels, metrics_base='binary', hm_preds=None, cls_lamda=0.1, threshold=0.5):
    assert metrics_base in ['binary', 'heatmap', 'combine'], 'Metric base is only one of these values [binary, heatmap, combine]'
    
    if cls_preds.shape[-1] > 1:
        cls_preds = cls_preds.softmax(dim=-1)
        cls_preds = cls_preds[:, -1]
    else:
        cls_preds = cls_preds.sigmoid()

    if metrics_base == 'combine':
        assert hm_preds is not None, 'Heatmap predict can not be None if metrics-base is combine'
        hm_preds = _sigmoid(hm_preds)
        hm_preds = torch.reshape(hm_preds, (hm_preds.shape[0], 1, -1))
        top_k = torch.topk(hm_preds, 10, -1).values
        mean_hm_preds = torch.mean(top_k, -1)
        cls_preds = cls_lamda*cls_preds + (1-cls_lamda)*mean_hm_preds

    labels = labels.cpu().numpy()
    cls_preds = cls_preds.cpu().numpy()
    fpr, tpr, thresholds = metrics.roc_curve(labels, cls_preds, pos_label=1)
    
    # AP metric
    ap = average_precision_score(labels, cls_preds)
    
    # AR metric
    ar = recall_score(labels, (cls_preds >= threshold).astype(int), average='macro')
    
    # mF1 metric
    mf1 = (ap*ar*2)/(ap+ar)
    
    return metrics.auc(fpr, tpr), ap, ar, mf1


def get_acc_mesure_func(task='binary'):
    if task == 'binary':
        return bin_calculate_acc
    elif task == 'heatmap':
        return hm_calculate_acc
    else:
        return hm_bin_calculate_acc
