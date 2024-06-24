import os
from pprint import pprint

import setproctitle

from _config import baseconfig as bconfig


class Config(bconfig):
    name = 'WHCF218-deepseg'
    vis_env = name
    network_name = "deepseg"
    setproctitle.setproctitle("%s" % name)

    gpu_id = "2"
    # scheduler
    scheduler = "step"
    warmup_steps = 400
    warmup_learning_rate_multiplier = 1 
    epoch = 700
    weight_decay = 2e-4
    lr_decay = 0.1
    # lr_decay = 0.9**(1/40)
    # lr_decay = 0.8**(1/12000)
    lr_decay_step_size = 50
    lr_decay_by_epoch = True
    lr = 1e-4
    end_lr = 1e-7
    momentum = 0.9
    use_adam = False  # Use Adam optimizer
    betas=(0.900, 0.999) 
    use_adamw = False 
    train_batch_size = 1
    val_batch_size = 1
    test_batch_size = 1
    # resize_width = 256
    center_crop_size = 256
    
    # loss function
    loss_mode = "focalloss"
    output_loss_weight=1.0
    fuse5_loss_weight=0.5
    fuse4_loss_weight=0.75
    fuse3_loss_weight=1.0
    fuse2_loss_weight=0.75
    fuse1_loss_weight=0.5
    acc_sigmoid_th = 0.5
    pos_pixel_weight = 1.0
    #focal loss
    GAMMA=2.0
    ALPHA=1.0 # 0.25

    #model parameters
    input_channels = 3
    num_classes = 1
    ngf = 64        # help='# of gen filters in the last conv layer'
    norm = "batch"  # help='instance normalization or batch normalization [instance | batch | none]'
    init_type = "xavier"  # help='network initialization [normal | xavier | kaiming | orthogonal]'
    init_gain = 0.02 # help='scaling factor for normal, xavier and orthogonal.'

    #guided_filter
    epsilon = 0.065 # help='eps = 1e-6*img_siz*img_siz'
    radius = 5

    # path
    IMG_SIZE=256
    use_dataaug = True
    tensorboardlog  = os.path.join("log", "tensorboard",name)
    train_data_path = 'datasets/crack500_train.txt'
    val_data_path   = 'datasets/crack500_val.txt'
    test_data_path  = 'datasets/WHCF218_val.txt'
    checkpoint_path = './checkpoints'
    log_path = './log/'+'visdom/'+name+'.txt'
    saver_path = os.path.join(checkpoint_path, name)
    max_save = 30
    save_condition = 0.2

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
