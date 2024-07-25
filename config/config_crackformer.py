import os
from pprint import pprint

import setproctitle

from _config import baseconfig as bconfig



class Config(bconfig):
    name = 'crack500-CrackFormer'
    vis_env = name
    network_name = "crackformer"
    setproctitle.setproctitle("%s" % name)

    gpu_id = '3'
    # scheduler
    scheduler = "exp"
    warmup_steps = 1
    warmup_learning_rate_multiplier = 1 
    epoch = 200
    weight_decay = 0.00001
    lr_decay = 0.8**(1/30)
    end_lr = 1e-7
    lr_decay_step_size = [20, 50, 100]
    lr_decay_by_epoch = True
    lr = 1e-4
    end_lr = 1e-7
    momentum = 0.9
    use_adam = True  # Use Adam optimizer
    use_adamw = False 
    betas=(0.900, 0.999)  
    train_batch_size = 1
    val_batch_size = 1
    test_batch_size = 1
    resize_width=None
    enlarge=False
    center_crop_size=None
    
    # loss function
    loss_mode = "bce"
    output_loss_weight=1.
    fuse5_loss_weight=1.
    fuse4_loss_weight=1.
    fuse3_loss_weight=1.
    fuse2_loss_weight=1.
    fuse1_loss_weight=1.
    acc_sigmoid_th = 0.5
    pos_pixel_weight = 1.
    #focal loss
    GAMMA=2.0
    ALPHA=0.25

    vis_train_loss_every = 500
    vis_train_acc_every  = 500
    vis_train_img_every  = 500
    vis_val_every        = 500

    # path
    IMG_SIZE = 512
    use_dataaug = True
    tensorboardlog  = os.path.join("log", "tensorboard",name)
    train_data_path = 'datasets/crack500_train.txt'
    val_data_path   = 'datasets/crack500_val.txt'
    test_data_path  = 'datasets/crack500_.txt'
    checkpoint_path = './checkpoints'
    log_path = './log/'+'visdom/'+name+'.txt'
    saver_path = os.path.join(checkpoint_path, name)
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

if __name__=="__main__":
    a = Config()
    print(a.log_path)