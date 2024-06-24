import sys

from torch import nn

sys.path.append('.')
import numpy as np
import torch

sys.path.append('./model')

from trainer._trainer import Trainer

from config.config_deepcrack import Config as cfg


class ModelTrainer(nn.Module, Trainer):
    def __init__(self, model):
        super(ModelTrainer, self).__init__()
        Trainer.__init__(self, cfg, model)

    def train_op(self, input, target):
        #--------------------warm up learning rate for pretraining ----------------#
        self.optimizer.zero_grad()
        pred_output, pred_fuse5, pred_fuse4, pred_fuse3, pred_fuse2, pred_fuse1, = self.model(input)
        output_loss = self.mask_loss(pred_output.view(-1, 1), target.view(-1, 1)) / cfg.train_batch_size
        fuse5_loss = self.mask_loss(pred_fuse5.view(-1, 1), target.view(-1, 1)) / cfg.train_batch_size
        fuse4_loss = self.mask_loss(pred_fuse4.view(-1, 1), target.view(-1, 1)) / cfg.train_batch_size
        fuse3_loss = self.mask_loss(pred_fuse3.view(-1, 1), target.view(-1, 1)) / cfg.train_batch_size
        fuse2_loss = self.mask_loss(pred_fuse2.view(-1, 1), target.view(-1, 1)) / cfg.train_batch_size
        fuse1_loss = self.mask_loss(pred_fuse1.view(-1, 1), target.view(-1, 1)) / cfg.train_batch_size
        total_loss = output_loss*cfg.output_loss_weight + fuse5_loss*cfg.fuse5_loss_weight + fuse4_loss*cfg.fuse4_loss_weight + fuse3_loss*cfg.fuse3_loss_weight + fuse2_loss*cfg.fuse2_loss_weight + fuse1_loss*cfg.fuse1_loss_weight
        total_loss.backward()
        self.optimizer.step()
        if not cfg.lr_decay_by_epoch:
            self.scheduler.step()
        self.log_loss = {
            'total_loss': total_loss.item(),
            'output_loss': output_loss.item(),
            'fuse5_loss': fuse5_loss.item(),
            'fuse4_loss': fuse4_loss.item(),
            'fuse3_loss': fuse3_loss.item(),
            'fuse2_loss': fuse2_loss.item(),
            'fuse1_loss': fuse1_loss.item()
        }
        return pred_output, pred_fuse5, pred_fuse4, pred_fuse3, pred_fuse2, pred_fuse1,

    def val_op(self, input, target):
        pred_output, pred_fuse5, pred_fuse4, pred_fuse3, pred_fuse2, pred_fuse1, = self.model(input)
        output_loss = self.mask_loss(pred_output.view(-1, 1), target.view(-1, 1)) / cfg.val_batch_size
        fuse5_loss = self.mask_loss(pred_fuse5.view(-1, 1), target.view(-1, 1)) / cfg.val_batch_size
        fuse4_loss = self.mask_loss(pred_fuse4.view(-1, 1), target.view(-1, 1)) / cfg.val_batch_size
        fuse3_loss = self.mask_loss(pred_fuse3.view(-1, 1), target.view(-1, 1)) / cfg.val_batch_size
        fuse2_loss = self.mask_loss(pred_fuse2.view(-1, 1), target.view(-1, 1)) / cfg.val_batch_size
        fuse1_loss = self.mask_loss(pred_fuse1.view(-1, 1), target.view(-1, 1)) / cfg.val_batch_size
        total_loss = output_loss*cfg.output_loss_weight + fuse5_loss*cfg.fuse5_loss_weight + fuse4_loss*cfg.fuse4_loss_weight + fuse3_loss*cfg.fuse3_loss_weight + fuse2_loss*cfg.fuse2_loss_weight + fuse1_loss*cfg.fuse1_loss_weight
        self.log_loss = {
            'total_loss':  total_loss.item(),
            'output_loss': output_loss.item(),
            'fuse5_loss':  fuse5_loss.item(),
            'fuse4_loss':  fuse4_loss.item(),
            'fuse3_loss':  fuse3_loss.item(),
            'fuse2_loss':  fuse2_loss.item(),
            'fuse1_loss':  fuse1_loss.item()
        }
        return pred_output, pred_fuse5, pred_fuse4, pred_fuse3, pred_fuse2, pred_fuse1,

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
        pred_mask = pred.squeeze(1).contiguous()
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
