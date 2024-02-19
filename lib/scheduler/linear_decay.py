#-*- coding: utf-8 -*-
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import _LRScheduler


class LinearDecayLR(_LRScheduler):
    def __init__(self, optimizer, n_epoch, start_decay, last_epoch=-1, booster=2):
        self.start_decay=start_decay
        self.n_epoch=n_epoch
        self.booster = booster
        super(LinearDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        last_epoch = self.last_epoch
        n_epoch = self.n_epoch
        b_lr = self.base_lrs[-1]

        if last_epoch > 0:
            try:
                cur_lr = self.get_last_lr()
            except:
                cur_lr = b_lr * self.booster
        start_decay = self.start_decay

        if last_epoch >= start_decay:
            lr = b_lr * self.booster - (b_lr * self.booster)/(n_epoch - start_decay) * (last_epoch - start_decay)
        else:
            if last_epoch < start_decay:
                lr = b_lr + (b_lr * self.booster - b_lr)/start_decay * last_epoch
            else:
                lr = cur_lr
                
        self._last_lr = lr
        print(f'Active Learning Rate --- {lr}')
        return [lr]
