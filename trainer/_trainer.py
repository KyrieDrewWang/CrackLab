import sys

from torch import nn

sys.path.append('.')
import torch

sys.path.append('./model')
from lossfunc.lossFunctions import cross_entropy_loss_RCF, BinaryFocalLoss
from lossfunc.diceloss import DiceLoss
from abc import ABC, abstractmethod
import numpy as np
from trainer.checkpointer import Checkpointer
from tools.visdom import Visualizer
from trainer.warmup_learning_rate import GradualWarmupScheduler
from matplotlib import pyplot as plt

def get_optimizer(model, lr, cfg):
    if cfg.use_adam:
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    elif cfg.use_adamw:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay, betas=cfg.betas)
    else:
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum)


class Trainer(ABC):
    """This class is an abstract base class (ABC) for _trainers.
    To create a subclass, you need to implement the following five functions:
    -- <__init__>:         initialize the class; first call Trainer.__init__(self, model, cfg).
    -- <train_op>:         training of the model.
    -- <val_op>:           validation of the model.
    -- <acc_op>:           accuracy calculation of the model.
    """
    def __init__(self, cfg, model): 
        
        self.model = model

        self.optimizer = get_optimizer(self.model, cfg.lr, cfg)
        
        #----------------------learning_rate_adjustment------------------#

        if cfg.scheduler == "exp":
            self._scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=cfg.lr_decay)
        elif cfg.scheduler == "step":
            self._scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=cfg.lr_decay_step_size, gamma=cfg.lr_decay)
        elif cfg.scheduler == "mulitsteps":
            self._scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=cfg.lr_decay_step_size, gamma=cfg.lr_decay)
        else:
            print("unknow scheduler mode")
        # -------------------- Loss --------------------- #
        
        if cfg.loss_mode == "bce-rcf":
            self.mask_loss = cross_entropy_loss_RCF
        elif cfg.loss_mode == "bce":
            self.mask_loss = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.cuda.FloatTensor([cfg.pos_pixel_weight]))
        elif cfg.loss_mode == "focalloss":
            self.mask_loss = BinaryFocalLoss(gamma=cfg.GAMMA, alpha=cfg.ALPHA)
        elif cfg.loss_mode == "diceloss":
            self.mask_loss == DiceLoss(1)
        else:
            print("unknow loss mode")
        
        self.vis = Visualizer(env=cfg.vis_env)
        self.vis.save_settings(save_path="log/visdom", save_log=True)
        self.saver = Checkpointer(cfg.name, cfg.saver_path, overwrite=False, verbose=True, timestamp=True,max_queue=cfg.max_save)
        self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=cfg.warmup_learning_rate_multiplier, total_epoch=cfg.warmup_steps, after_scheduler=self._scheduler)
        self.iters = 0
        self.log_loss = {}
        self.log_acc = {}

    @abstractmethod
    def train_op(self, input, target):
        pass

    @abstractmethod
    def val_op(self, input, target):
        pass

    @abstractmethod
    def acc_op(self, pred, target):
        pass

    
    def get_statistics(self, pred, gt):
        tp = int(torch.sum((pred==1) & (gt==1)))
        fp = int(torch.sum((pred==1) & (gt==0)))
        fn = int(torch.sum((pred==0) & (gt==1)))
        return [tp,fp,fn]

    # ODS calculation
    def cal_ods_metrics(self, pred_list, gt_list, thresh_step=0.01):
        batch_accuracy = []
        for thresh in np.arange(0.0, 1.0, thresh_step):
            statistics = []
            for pred, gt in zip(pred_list, gt_list):
                gt_img = gt.type(torch.int8)
                pred_img = (pred > thresh).type(torch.int8)
                # calculate each image
                statistics.append(self.get_statistics(pred_img, gt_img))
            # get tp, fp, fn
            tp = np.sum([v[0] for v in statistics])
            fp = np.sum([v[1] for v in statistics])
            fn = np.sum([v[2] for v in statistics])
            # calculate precision
            p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
            # calculate recall
            r_acc = tp / (tp + fn + 1e-6)
            # calculate f-score
            batch_accuracy.append([thresh, p_acc, r_acc, 2 * p_acc * r_acc / (p_acc + r_acc + 1e-6)])
        return batch_accuracy    

    # OIS calculation
    def cal_ois_metrics(self,pred_list, gt_list, thresh_step=0.01):
        batch_acc_all = []
        for pred, gt in zip(pred_list, gt_list):
            statistics = []
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
                statistics.append([thresh, f1])
            max_f = np.amax(statistics, axis=0)
            batch_acc_all.append(max_f[1])
        return np.mean(batch_acc_all)

if __name__ == "__main__":
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=100)
    after =  torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=50, after_scheduler=after)
    lrs = []

    for i in range(100):
        optimizer.step()
        lrs.append(optimizer.param_groups[0]["lr"])
    #     print("Factor = ",0.1 if i!=0 and i%2!=0 else 1," , Learning Rate = ",optimizer.param_groups[0]["lr"])
        scheduler.step()

    plt.plot(range(100),lrs)
    plt.savefig("test.png")