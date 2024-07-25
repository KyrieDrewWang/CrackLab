import os
from pprint import pprint

import setproctitle
import sys
sys.path.append('.')
from config._config import baseconfig as bconfig


class Config(bconfig):
    name = 'CrackLS315-DeformConvCrack-offset2-CBAMup'
    vis_env = name 
    network_name = "DeformConvCrack"
    desc="basic DeformConvCrack settings"
    setproctitle.setproctitle("%s" % name)

    gpu_id = '8'
    # scheduler
    scheduler = "exp"
    # warmup_steps = 1500
    warmup_steps = 5
    epoch = 200
    weight_decay = 0.00001
    lr = 1e-3
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
    train_batch_size = 1
    val_batch_size = 1
    test_batch_size = 1

    # loss function
    loss_mode = "bce"
    output_loss_weight=1.
    fuse5_loss_weight=0.
    fuse4_loss_weight=0.
    fuse3_loss_weight=0.
    fuse2_loss_weight=0.
    fuse1_loss_weight=0.
    acc_sigmoid_th = 0.5
    pos_pixel_weight = 1.
    #focal loss
    GAMMA=2.0
    ALPHA=0.25

    #args for interimage model
    IMG_SIZE=512
    CORE_OP='DCNv3'
    CHANNELS=112
    DEPTHS=[4, 4, 21, 4]    # [5, 5, 22, 5]
    GROUPS=[7, 14, 28, 56]  # [10, 20, 40, 80]
    MLP_RATIO=4.
    DROP_RATE=0.1
    DROP_PATH_RATE=0.4
    DROP_PATH_TYPE='linear'
    ACT_LAYER='GELU'
    NORM_LAYER='LN'
    LAYER_SCALE=1.0
    POST_NORM=True
    WITH_CP=False
    OFFSET_SCALE=2.0
    PRETRAIN_PATH="pretrained_model/internimage_b_1k_224.pth" # 'pretrained_model/internimage_l_22k_192to384.pth'
    
    # args for interimage model structure

    AG = False
    AGD = True
    
    SIDEOUT = False
    UPCBAM = False
    
    CBAMUP = True
    
    CBAMDECODER = False

    SE_DOWN = False
    LOCAL_ENHANCE_FFN = False
    CBAMAGG = False
    RECTIFICATOIN = False
    PIXEL_SHUFFLE_DOWN = False
    PIXEL_SHUFFLE_UP = False
    STEMPRO = False
    LAMBDA_UP = False
    PA = False
    # path
    use_dataaug = True
    tensorboardlog  = os.path.join("log", "tensorboard",name)
    train_data_path = 'datasets/CrackLS315_train.txt'
    val_data_path   = 'datasets/CrackLS315_val.txt'
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