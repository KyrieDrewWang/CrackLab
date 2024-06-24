import sys
sys.path.append('.')
import numpy as np
import torch
import torch.nn as nn
sys.path.append('./model')
from trainer._trainer import Trainer

class ModelTrainer(nn.Module, Trainer):
    def __init__(self, model, cfg):
        super(ModelTrainer, self).__init__()
        Trainer.__init__(self, cfg, model)
        self.cfg = cfg
        self.ois_merits = []
        self.ods_merits = []
    def train_op(self, input, target):
        #--------------------warm up learning rate for pretraining ----------------#
        self.optimizer.zero_grad()
        pred_output, pred_fuse5, pred_fuse4, pred_fuse3, pred_fuse2, pred_fuse1= self.model(input)
        output_loss = self.mask_loss(pred_output.view(-1, 1), target.view(-1, 1)) / self.cfg.train_batch_size
        fuse5_loss = self.mask_loss(pred_fuse5.view(-1, 1), target.view(-1, 1)) / self.cfg.train_batch_size
        fuse4_loss = self.mask_loss(pred_fuse4.view(-1, 1), target.view(-1, 1)) / self.cfg.train_batch_size
        fuse3_loss = self.mask_loss(pred_fuse3.view(-1, 1), target.view(-1, 1)) / self.cfg.train_batch_size
        fuse2_loss = self.mask_loss(pred_fuse2.view(-1, 1), target.view(-1, 1)) / self.cfg.train_batch_size
        fuse1_loss = self.mask_loss(pred_fuse1.view(-1, 1), target.view(-1, 1)) / self.cfg.train_batch_size
        total_loss = output_loss*self.cfg.output_loss_weight + fuse5_loss*self.cfg.fuse5_loss_weight + fuse4_loss*self.cfg.fuse4_loss_weight + fuse3_loss*self.cfg.fuse3_loss_weight + fuse2_loss*self.cfg.fuse2_loss_weight + fuse1_loss*self.cfg.fuse1_loss_weight
        total_loss.backward()
        self.optimizer.step()
        if not self.cfg.lr_decay_by_epoch:
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
        return pred_output, pred_fuse5, pred_fuse4, pred_fuse3, pred_fuse2, pred_fuse1

    def val_op(self, input, target):
        pred_output, pred_fuse5, pred_fuse4, pred_fuse3, pred_fuse2, pred_fuse1 = self.model(input)
        output_loss = self.mask_loss(pred_output.view(-1, 1), target.view(-1, 1)) / self.cfg.val_batch_size
        fuse5_loss = self.mask_loss(pred_fuse5.view(-1, 1), target.view(-1, 1)) / self.cfg.val_batch_size
        fuse4_loss = self.mask_loss(pred_fuse4.view(-1, 1), target.view(-1, 1)) / self.cfg.val_batch_size
        fuse3_loss = self.mask_loss(pred_fuse3.view(-1, 1), target.view(-1, 1)) / self.cfg.val_batch_size
        fuse2_loss = self.mask_loss(pred_fuse2.view(-1, 1), target.view(-1, 1)) / self.cfg.val_batch_size
        fuse1_loss = self.mask_loss(pred_fuse1.view(-1, 1), target.view(-1, 1)) / self.cfg.val_batch_size
        total_loss = output_loss*self.cfg.output_loss_weight + fuse5_loss*self.cfg.fuse5_loss_weight + fuse4_loss*self.cfg.fuse4_loss_weight + fuse3_loss*self.cfg.fuse3_loss_weight + fuse2_loss*self.cfg.fuse2_loss_weight + fuse1_loss*self.cfg.fuse1_loss_weight
        self.log_loss = {
            'total_loss':  total_loss.item(),
            'output_loss': output_loss.item(),
            'fuse5_loss':  fuse5_loss.item(),
            'fuse4_loss':  fuse4_loss.item(),
            'fuse3_loss':  fuse3_loss.item(),
            'fuse2_loss':  fuse2_loss.item(),
            'fuse1_loss':  fuse1_loss.item()
        }
        return pred_output, pred_fuse5, pred_fuse4, pred_fuse3, pred_fuse2, pred_fuse1

    def acc_op(self, pred, target):
        mask = target   #.clone()
        pred = torch.sigmoid(pred)

        pred_list = [i for i in pred]
        mask_list = [j for j in mask]
        
        batch_ods = self.cal_ods_metrics(pred_list, mask_list, 0.01)
        batch_ois = self.cal_ois_metrics(pred_list, mask_list, 0.01)
        max_f = np.amax(batch_ods, axis=0)
        pred[pred > self.cfg.acc_sigmoid_th] = 1
        pred[pred <= self.cfg.acc_sigmoid_th] = 0
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

    def val_OIS_cal(self, pred, gt, thresh_step=0.01):
        pred = torch.sigmoid(pred)
        statistic = []
        for thresh in np.arange(0.0, 1.0, thresh_step):
            gt_img = gt.type(torch.int8)
            pred_img = (pred > thresh).type(torch.int8)
            tp, fp, fn = self.get_statistics(pred_img, gt_img)
            p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
            r_acc = tp / (tp + fn + 1e-6)
            if p_acc + r_acc == 0:
                f1 = 0
            else:
                f1 = 2 * p_acc * r_acc / (p_acc + r_acc)
            statistic.append([thresh, f1])
        max_f = np.amax(statistic, axis=0) 
        self.ois_merits.append(max_f[1])
        
    def cal_ois_scores(self,):
        return np.mean(self.ois_merits)
    
    def val_ODS_cal(self, pred, gt, thresh=0.9):
        pred = torch.sigmoid(pred)
        gt_img = gt.type(torch.int8)
        pred_img = (pred > thresh).type(torch.int8)
        self.ods_merits.append(self.get_statistics(pred_img, gt_img))
        
    def cal_ods_scores(self,):
        tp = np.sum([v[0] for v in self.ods_merits])
        fp = np.sum([v[1] for v in self.ods_merits])
        fn = np.sum([v[2] for v in self.ods_merits])
        
        p_acc = 1.0 if tp == 0 and fp == 0 else tp/(tp + fp)
        r_acc = tp / (tp + fn + 1e-6)
        merits = [p_acc, r_acc, 2*p_acc*r_acc / (p_acc + r_acc +1e-6)]
        return merits[2]
    
