#-*- coding: utf-8 -*-
import os
import logging
from types import MethodType
from datetime import datetime

import torch
import torch.nn.functional as F
from package_utils.utils import make_dir


LOG_DIR = 'logs/{}'.format(datetime.today().strftime('%d-%m-%Y'))
make_dir(LOG_DIR)


class Logger():
    def __init__(self, task='training'):
        super().__init__()
        self.logger = logging.getLogger('')
        self.logger.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler('{}/{}.log'.format(LOG_DIR, task))
        stream_handler = logging.StreamHandler()
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)
        self.logger.epochInfor = MethodType(self.epochInfor, self.logger)
        self.info = self.logger.info
    
    def epochInfor(self, 
                   epoch, 
                   idx,
                   length,
                   batch_time,
                   speed, 
                   data_time,
                   losses,
                   acc,
                   loss_cls=None,
                   **kwargs):
        msg = 'Epoch: [{0}][{1}/{2}]\t' \
              'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
              'Speed {speed:.1f} samples/s\t' \
              'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
              'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
              'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch, idx, length,
                batch_time=batch_time,
                speed=speed,
                data_time=data_time,
                loss=losses,
                acc=acc
              )
        if loss_cls is not None:
            msg += '\t Cls Loss: {loss_cls:.5f}'.format(loss_cls=loss_cls)
        
        for k,v in kwargs.items():
            if v is not None:
                msg += f'\t {k}: {v:.5f}'
        
        self.logger.info(msg)


def board_writing(writer, loss, acc, iterations, dataset='Train'):
    writer.add_scalar(
        '{}/loss'.format(dataset), loss, iterations)
    writer.add_scalar(
        '{}/acc'.format(dataset), acc, iterations)


def debug_writing(writer, outputs, labels, inputs, iterations):
    tmp_tar = torch.unsqueeze(labels.cpu().data[0], dim=1)
    # tmp_out = torch.unsqueeze(outputs.cpu().data[0], dim=1)

    tmp_inp = inputs.cpu().data[0]
    tmp_inp[0] += 0.406
    tmp_inp[1] += 0.457
    tmp_inp[2] += 0.480

    tmp_inp[0] += torch.sum(F.interpolate(tmp_tar, scale_factor=4, mode='bilinear'), dim=0)[0]
    tmp_inp.clamp_(0, 1)

    writer.add_image('Data/input', tmp_inp, iterations)
