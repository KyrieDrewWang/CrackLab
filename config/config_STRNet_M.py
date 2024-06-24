import os
from pprint import pprint

import setproctitle
import sys
sys.path.append('.')
from config._config import baseconfig as bconfig


class Config(bconfig):
    name = 'CrackLS315-STRNet_M'
    vis_env = name 
    network_name = "STRNet_M"
    desc="best model settings"
    setproctitle.setproctitle("%s" % name)

    gpu_id = '3'
    # scheduler
    scheduler = "exp"
    warmup_steps = 10
    epoch = 500
    weight_decay = 0.00001
    lr = 1e-4
    # lr_decay = 0.90**(1/12000) 
    lr_decay = 0.9**(1/45)  
    lr_decay_step_size = [300, 400, 450]
    lr_decay_by_epoch = True

    warmup_learning_rate_multiplier = 1
    end_lr = 1e-7
    momentum = 0.9
    use_adam = False  # Use Adam optimizer
    use_adamw = True   
    betas=(0.900, 0.999)  
    
    #dataloader
    train_batch_size = 1
    val_batch_size = 1
    test_batch_size = 1
    # resize_width=224
    # enlarge=False
    # center_crop_size=224
    
    # loss function
    loss_mode = "bce"
    output_loss_weight=1.
    fuse5_loss_weight=0.
    fuse4_loss_weight=0.
    fuse3_loss_weight=0.
    fuse2_loss_weight=0.
    fuse1_loss_weight=0.
    acc_sigmoid_th = 0.5
    pos_pixel_weight = 1
    #focal loss
    GAMMA=2.0
    ALPHA=0.25

    # args for SDDNet_M:
    CHANNEL_IN=3
    CHANNEL_OUT=1

    # path
    IMG_SIZE = 512
    use_dataaug = True
    tensorboardlog  = os.path.join("log", "tensorboard",name)
    train_data_path = 'datasets/WHCF218_train.txt'
    val_data_path   = 'datasets/WHCF218_val.txt'
    test_data_path  = 'datasets/CrackLS315_val.txt'
    checkpoint_path = './checkpoints'
    log_path = './log/'+'visdom/'+name+'.txt'
    saver_path = os.path.join(checkpoint_path, name)  # where checkpointer is saved 
    max_save = 30
    save_condition = 0.5
    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        a = {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}
        a.update({k: getattr(bconfig, k) for k , _ in bconfig.__dict__.items()})
        return a

    def show(self):
        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

if __name__ == "__main__":
    c = Config()
    print(c.vis_train_loss_every)