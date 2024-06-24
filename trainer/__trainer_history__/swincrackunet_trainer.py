import sys

sys.path.append('.')
import numpy as np
import torch
import torch.nn as nn

from config.config_swincrackunet import Config as cfg

sys.path.append('./model')
from trainer._trainer import Trainer


class ModelTrainer(nn.Module, Trainer):
    def __init__(self, model):
        super(ModelTrainer, self).__init__()
        Trainer.__init__(self, cfg, model)

    def train_op(self, input, target):
        #--------------------warm up learning rate for pretraining ----------------#
        self.optimizer.zero_grad()
        pred_output = self.model(input)
        total_loss = self.mask_loss(pred_output, target)  
        total_loss.backward()
        self.optimizer.step()
        if not cfg.lr_decay_by_epoch:
            self.scheduler.step()
        self.log_loss = {
            'total_loss': total_loss.item(),
        }
        return pred_output

    def val_op(self, input, target):
        pred_output = self.model(input)
        total_loss = self.mask_loss(pred_output, target)  
        self.log_loss = {
            'total_loss': total_loss.item(),
        }
        return pred_output

    def acc_op(self, pred, target):
        mask = target.clone()
        pred = torch.sigmoid(pred.clone())

        pred_list = [i for i in pred]
        mask_list = [j for j in mask]
        
        batch_prf = self.cal_prf_metrics(pred_list, mask_list, 0.01)
        batch_ois = self.cal_ois_metrics(pred_list, mask_list, 0.01)
        max_f = np.amax(batch_prf, axis=0)

        pred[pred > cfg.acc_sigmoid_th] = 1
        pred[pred <= cfg.acc_sigmoid_th] = 0
        pred_mask = pred.contiguous()
        mask_acc = pred_mask.eq(mask.view_as(pred_mask)).sum().item() / mask.numel()
        mask_pos_acc = pred_mask[mask > 0].eq(mask[mask > 0].view_as(pred_mask[mask > 0])).sum().item() / (mask[
            mask > 0].numel() + 1e-6)
        mask_neg_acc = pred_mask[mask < 1].eq(mask[mask < 1].view_as(pred_mask[mask < 1])).sum().item() / mask[
            mask < 1].numel()

        self.log_acc = {
            'pred_acc': mask_acc,
            'pred_pos_acc': mask_pos_acc,
            'pred_neg_acc': mask_neg_acc,
            'pred_ODS': max_f[3],
            'pred_OIS': batch_ois,
        }
