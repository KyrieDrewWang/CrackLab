import os
from pprint import pprint

import setproctitle
import sys
sys.path.append('.')
from config._config import baseconfig as bconfig


class Config(bconfig):
    name = 'CrackLS315-SwinUperCrack'
    vis_env = name 
    network_name = "swinupercrack"
    setproctitle.setproctitle("%s" % name)

    gpu_id = '5'
    # scheduler
    scheduler = "exp"
    warmup_steps = 10
    warmup_learning_rate_multiplier = 1
    epoch = 500
    weight_decay = 0.00005
    lr_decay = 0.9**(1/45)  # 0.90**(1/12000)
    # lr_decay_step_size = [300, 400, 450]
    lr_decay_by_epoch = True
    lr = 1e-4
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
    pos_pixel_weight = 1.0
    #focal loss
    GAMMA=2.0
    ALPHA=0.25

    #args for swin model
    IMG_SIZE=512
    PATCH_SIZE = 4
    IN_CHANS = 3
    EMBED_DIM = 128
    DEPTHS =  [2, 2, 18, 2]
    NUM_HEADS = [4, 8, 16, 32]
    WINDOW_SIZE = 7
    MLP_RATIO = 4.
    QKV_BIAS = True
    QK_SCALE = None
    DROP_RATE = 0.3
    ATTN_DROP_RATE = 0.
    DROP_PATH_RATE = 0.5
    APE = False
    PATCH_NORM = True
    USE_CHECKPOINT = False
    PRETRAIN_CKPT = "./pretrained_model/swin_base_patch4_window7_224.pth"
    NUM_CLASSES = 1

    #args for swin decoder
    DE_DEPTHS = [2, 18, 2, 2]   
    DE_NUM_HEADS = [32, 16, 8, 4]  # [3,6,12,24] 
    DE_DROP_RATE = 0.3
    DE_ATTN_DROP_RATE = 0.
    DE_DROP_PATH_RATE = 0.5

    # introducing convolution
    use_convEmbd = True
    use_SC = False
    
    # path
    use_dataaug = True
    tensorboardlog  = os.path.join("log", "tensorboard",name)
    train_data_path = 'datasets/CrackLS315_src_train.txt'
    val_data_path   = 'datasets/CrackLS315_src_val.txt'
    test_data_path  = 'datasets/CrackLS315_src_test.txt'
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
    c.show()