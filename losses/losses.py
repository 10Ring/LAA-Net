#-*- coding: utf-8 -*-
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import BCELoss, CrossEntropyLoss
import torch.nn.functional as F
from torch.nn.functional import binary_cross_entropy

from .builder import LOSSES


def _sigmoid(hm):
    x = hm
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return y


def _avg_sigmoid(hm):
    if hm.dim() == 4:
        x = torch.mean(hm, [2, 3])
    else:
        x = hm
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return y


def f_cstency(cstency_hm_preds, cstency_hm_gt):
    # Heatmap here that is original is returned from model without any modification
    cstency_matrix = torch.zeros_like(cstency_hm_gt).cuda()
    b_size = cstency_hm_preds.size(0)
    cst_hm_dim = cstency_hm_preds.size(1)
    cst_hm_h = cstency_hm_preds.size(2)
    cst_hm_w = cstency_hm_preds.size(3)
    
    indices_ = cstency_hm_gt.view(b_size, -1).argmax(dim=-1)
    # anchor_indices = (np.arange(b_size), :, indices_//b_size, indices_%b_size)
    cstency_matrix_ = torch.matmul(
        cstency_hm_preds.view(b_size, cst_hm_dim, -1)[np.arange(b_size), :, indices_].view(b_size, 1, cst_hm_dim),
        cstency_hm_preds.view(b_size, cst_hm_dim, -1)
    )
    cstency_matrix_ = cstency_matrix_.view(b_size, cstency_hm_gt.size(1), cst_hm_h, cst_hm_w) / math.sqrt(cst_hm_dim)
    cstency_matrix = cstency_matrix_.sigmoid_()
    
    # for i in range(b_size):
    #     anchor_pos = torch.where(cstency_heatmap_gt[i, 0] == cstency_heatmap_gt[i, 0].max())
    #     for k in range(cst_hm_h):
    #         for l in range(cst_hm_w):
    #             cstency_matrix[i, 0, k, l] = \
    #                 torch.matmul(cstency_heatmap_preds[i, :, k, l], cstency_heatmap_preds[i, :, anchor_pos[0][0], anchor_pos[1][0]])
    #             cstency_matrix[i, 0, k, l] = (cstency_matrix[i, 0, k, l] / math.sqrt(cst_hm_dim)).sigmoid_()
    return cstency_matrix


def _neg_pos_loss(hm_pred, hm_gt):
    pos_idxes = hm_gt > 0
    neg_idxes = ~pos_idxes
    batch_size = hm_gt.size(0)
    neg_pos_gt, neg_pos_pred = torch.zeros(batch_size, 1, dtype=torch.float64).cuda(), \
                                    torch.zeros(batch_size, 1, dtype=torch.float64).cuda()
    hm_pred_ = torch.squeeze(torch.clone(hm_pred))
    
    for i in range(batch_size):
        neg_pos_gt[i] = torch.sum(hm_gt[i][pos_idxes[i,:,:]]) - torch.sum(hm_gt[i][neg_idxes[i,:,:]])
        neg_pos_pred[i] = torch.sum(hm_pred_[i][pos_idxes[i,:,:]]) - torch.sum(hm_pred_[i][neg_idxes[i,:,:]])
        
    return torch.abs(neg_pos_pred), torch.abs(neg_pos_gt)


def _neg_loss(pred, gt, epsilon=0.35, noise_distribution=0.2, alpha=0.25):
    ''' Modified focal loss. Exactly the same as CornerNet.
    Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
    '''
    fine_grained_ratio = [0.5, 1.5, 0.5, 1.5]
    loss = 0
    pos_inds = gt.eq(1.0).float()
    neg_inds = gt.lt(1.0).float()
    b_size = gt.shape[0] 

    # non_zero_inds = torch.nonzero(pos_inds)
    # rnd_sples = np.random.randint(0, len(non_zero_inds), b_size*8)
    # non_zero_rnd = non_zero_inds[rnd_sples]
    # non_zero_rnd = (non_zero_rnd[:, 0], non_zero_rnd[:, 1], non_zero_rnd[:, 2], non_zero_rnd[:, 3])

    # rnd_pos_inds = torch.zeros_like(pos_inds)
    # rnd_pos_inds[non_zero_rnd] = 1

    neg_weights = torch.pow(1 - gt, 4)

    # pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds * alpha
    pos_loss = (1 - epsilon) * torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    pos_loss_noise = epsilon * torch.log(pred) * torch.pow(1 - pred, 2) * noise_distribution * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds * neg_weights

    pos_loss = pos_loss + pos_loss_noise

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    loss *= alpha
    return loss


def _distance_hm_cls_loss(cos_sim_ins, hm_preds, hm_gts, label_preds, label_gts, alpha=0.25):
    b_size = hm_preds.size(0)
    hm_preds = hm_preds.view(b_size, -1)
    hm_gts = hm_gts.view(b_size, -1)
    pos_hm_loss = 0.0
    neg_hm_loss = 0.0

    for i in range(0, b_size//2):
        for j in range(0, b_size//2):
            pos_hm_loss += (1/2) * (1 - cos_sim_ins(hm_preds[i], hm_preds[j]))
            neg_hm_loss += (1/2) * (1 - cos_sim_ins(hm_preds[i], hm_preds[j + b_size//2]))

    cos_loss = pos_hm_loss/((b_size//2)**2) - neg_hm_loss/((b_size//2)**2)
    cos_loss = cos_loss * alpha
    return cos_loss


@LOSSES.register_module()
class BaseLoss(nn.Module):
    def __init__(self, cfg, **kwargs):
        self.cfg = cfg
        super().__init__()
        
        for k,v in kwargs.items():
            if v is not None:
                self.__setattr__(k, v)
        # Critetion ins
        self.mse_critetion = nn.MSELoss(reduction=self.cfg.mse_reduction)
        self.bce_critetion = nn.BCELoss(reduction=self.cfg.ce_reduction) # For Binary Cross Entropy Loss
        self.ce_critetion = CrossEntropyLoss(reduction=self.cfg.ce_reduction) # For Cross Entropy Loss in general

        # Lambda coefs
        self.offset_lmda = self.cfg.offset_lmda
        self.cls_lmda = self.cfg.cls_lmda
        self.dst_hm_cls_lmda = self.cfg.dst_hm_cls_lmda
        self.hm_lmda = self.cfg.hm_lmda
        self.cstency_lmda = self.cfg.cstency_lmda

        # Others
        self.cos_sim_ins = nn.CosineSimilarity(dim=0, eps=1e-6)

    def _offset_loss(self, preds, gts, apply_filter=False):
        loss = 0
        coefs = gts.gt(0).float() if apply_filter else 1
        n_coefs = coefs.float().sum()
        
        loss = 0.5 * self.mse_critetion(preds * coefs, gts * coefs)
        loss /= (n_coefs + 1e-6)
        loss *= self.offset_lmda
        return loss
    
    def _cls_loss(self, preds, gts):
        loss = 0
        loss = self.bce_critetion(preds, gts)
        loss *= self.cls_lmda
        return loss
    
    def _consistency_loss(self, preds, gts):
        loss = torch.zeros(1).cuda()
        encode_preds = f_cstency(preds, gts)
        total_pixels = gts.size(2) * gts.size(3)
        loss = self.bce_critetion(encode_preds.view(-1, 1), gts.view(-1, 1))
        loss *= self.cstency_lmda
        return loss.sum()
    

@LOSSES.register_module()
class BinaryCrossEntropy(nn.Module):
    def __init__(self, reduction='mean'):
        super(BinaryCrossEntropy, self).__init__()
        self.reduction = reduction
        self.bce = BCELoss(reduction=self.reduction)
        
    def __call__(self, pred, y):
        return self.bce(pred, y)


@LOSSES.register_module()
class CombinedFocalLoss(BaseLoss):
    '''nn.Module warpper for focal loss'''
    def __init__(self, 
                 cfg,
                 use_target_weight, 
                 **kwargs):
        super(CombinedFocalLoss, self).__init__(cfg, **kwargs)
        self.neg_loss = _neg_loss
        self.use_target_weight = use_target_weight
        
    def forward(self, 
                hm_outputs, hm_targets, 
                cls_preds, cls_gts, 
                offset_preds=None,
                offset_gts=None,
                cstency_preds=None,
                cstency_gts=None,
                target_weight=None):
        loss_return = {}
        hm_outputs_ = torch.clone(hm_outputs)
        hm_outputs_ = _sigmoid(hm_outputs_)
        if hm_targets.dim() == 3:
            hm_targets = torch.unsqueeze(hm_targets, 1)
        
        loss_hm = self.neg_loss(hm_outputs_, hm_targets, alpha=self.hm_lmda)
        loss_return['hm'] = loss_hm
        loss_return['cls'] = self._cls_loss(cls_preds, cls_gts)
        
        if self.dst_hm_cls_lmda > 0:
            loss_return['dst_hm_cls'] = _distance_hm_cls_loss(self.cos_sim_ins, 
                                                              hm_outputs, 
                                                              hm_targets, 
                                                              cls_preds, 
                                                              cls_gts, 
                                                              alpha=self.dst_hm_cls_lmda)
        
        if self.offset_lmda > 0 and offset_preds is not None:
            loss_return['offset'] = self._offset_loss(offset_preds, 
                                                      offset_gts, 
                                                      apply_filter=True)
        
        if self.cstency_lmda > 0 and cstency_preds is not None:
            loss_return['cstency'] = self._consistency_loss(cstency_preds, 
                                                            cstency_gts)
        
        return loss_return


@LOSSES.register_module()
class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight, reduction='mean'):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction=reduction)
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight=None):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight and target_weight is not None:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


@LOSSES.register_module()
class CombinedLoss(nn.Module):
    def __init__(self, use_target_weight, cls_lmda=0.2, dst_lmda=0.2, reduction='mean', dist_cal=True, cls_cal=True, **kwargs):
        super(CombinedLoss, self).__init__()
        self.criterion_cls = BinaryCrossEntropy(reduction=reduction)
        self.criterion_hm = JointsMSELoss(use_target_weight=use_target_weight, reduction=reduction)
        self.critetion_dst = nn.MSELoss(reduction=reduction)
        self.use_target_weight = use_target_weight
        self.cls_lmda = cls_lmda if cls_cal else 0
        self.dst_lmda = dst_lmda if dist_cal else 0
        self.dst_loss = _neg_pos_loss
        self.dist_cal = dist_cal
        self.cls_cal = cls_cal

    def forward(self, hm_outputs, hm_targets, cls_preds, cls_gts, target_weight=None):
        loss_return = {}
        loss_hm = self.criterion_hm(hm_outputs, hm_targets)
        loss_return['hm'] = loss_hm

        loss_cls = self.criterion_cls(cls_preds, cls_gts)
        loss_return['cls'] = loss_cls
        
        # Defining negative and positive distance Loss which try to make the distance between the 2 elements certain range
        if self.dist_cal:
            neg_pos_pred, neg_pos_gt = self.dst_loss(hm_outputs, hm_targets)
            neg_pos_loss = 0.5 * self.critetion_dst(neg_pos_pred, neg_pos_gt)
            loss_return['dst_pos_neg'] = neg_pos_loss
        return loss_return


@LOSSES.register_module()
class CombinedHeatmapBinaryLoss(nn.Module):
    def __init__(self, use_target_weight, cls_lmda=0.2, reduction='mean', cls_cal=True, **kwargs):
        super(CombinedHeatmapBinaryLoss, self).__init__()
        self.criterion_cls = BinaryCrossEntropy(reduction=reduction)
        self.criterion_hm = BinaryCrossEntropy(reduction=reduction)
        self.use_target_weight = use_target_weight
        self.cls_lmda = cls_lmda if cls_cal else 0
        self.cls_cal = cls_cal

    def forward(self, hm_outputs, hm_targets, cls_preds, cls_gts, target_weight=None):
        batch_size = hm_outputs.size(0)
        hm_targets = hm_targets[:,:,:,0]
        hm_h = hm_outputs.size(2)
        hm_w = hm_outputs.size(3)
        total_pixels = hm_h * hm_w
        loss_hm = torch.zeros(1).cuda()
        hm_outputs_ = torch.clone(hm_outputs)
        hm_outputs_ = _sigmoid(hm_outputs_)
        
        for i in range(hm_h):
            for j in range(hm_w):
                loss_hm_ = self.criterion_hm(hm_outputs_[:,:,i,j], torch.unsqueeze(hm_targets[:,i,j], 1))
                loss_hm += loss_hm_
        
        loss_hm = loss_hm / total_pixels
        loss_return = {}
        loss_return['hm'] = loss_hm
        
        loss_cls = self.criterion_cls(cls_preds, cls_gts)
        loss_return['cls'] = loss_cls
        return loss_return


@LOSSES.register_module()
class CombinedPolyLoss(nn.Module):
    """
    PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions
    """
    def __init__(self, use_target_weight, epsilon=2.0, cls_lmda=0.05, reduction='mean', cls_cal=True, **kwargs):
        super(CombinedPolyLoss, self).__init__()
        self.cls_critetion = BinaryCrossEntropy(reduction=reduction)
        self.use_target_weight = use_target_weight
        self.epsilon = epsilon
        self.cls_lmda = cls_lmda if cls_cal else 0
        self.reduction = reduction
        self.cls_cal = cls_cal

    def forward(self, hm_outputs, hm_targets, cls_preds, cls_gts):
        batch_size = hm_outputs.size(0)
        n_classes = hm_outputs.size(1)
        hm_h = hm_outputs.size(2)
        hm_w = hm_outputs.size(3)
        total_pixels = hm_h * hm_w
        poly_loss = torch.zeros(batch_size, 1).cuda()
        hm_outputs_ = _sigmoid(hm_outputs)
        
        for i in range(hm_h):
            for j in range(hm_w):
                ce = binary_cross_entropy(hm_outputs_[:,:,i,j], torch.unsqueeze(hm_targets[:,i,j], -1), reduction='none')
                pt = hm_outputs_[:,:,i,j]
                pt = torch.squeeze(pt)
                pt = torch.where(hm_targets[:,i,j] > 0, pt, 1-pt)
                poly_loss += (ce + self.epsilon * (1.0 - torch.unsqueeze(pt, -1)))

        if self.reduction == 'mean':
            poly_loss = poly_loss.sum()/total_pixels/batch_size
        else:
            poly_loss = poly_loss.sum()
        loss_return = {}
        loss_return['hm'] = poly_loss
        
        loss_cls = self.cls_critetion(cls_preds, cls_gts)
        loss_return['cls'] = loss_cls * self.cls_lmda
        return loss_return
 