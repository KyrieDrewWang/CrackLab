import os
from pprint import pprint

import setproctitle

from _config import baseconfig as bconfig


class Config(bconfig):
    name = 'SwinCrackUnet-base'
    vis_env = 'SwinCrackUnet-base'
    setproctitle.setproctitle("%s" % name)

    gpu_id = '4'
    # scheduler
    scheduler = "exp"
    warmup_steps = 20
    warmup_learning_rate_multiplier = 1 
    epoch = 500
    lr = 1e-3
    end_lr = 1e-7
    lr_decay = 0.8**(1/30)
    lr_decay_step_size = [100, 200, 300]
    weight_decay = 0.00000
    momentum = 0.9
    use_adam = True  # Use Adam optimizer
    use_adamw = False 
    train_batch_size = 1
    val_batch_size = 1
    test_batch_size = 1

    # loss function
    loss_mode = "bce"
    acc_sigmoid_th = 0.5
    pos_pixel_weight = 1
    #focal loss
    GAMMA=2.0
    ALPHA=0.25

    #args for swin model
    IMG_SIZE=512
    PATCH_SIZE = 4
    IN_CHANS = 3
    EMBED_DIM = 96
    DEPTHS =  [2,2,6,6,2]       #[2, 2, 18, 2]
    NUM_HEADS = [3,6,12,12,24]  #[4, 8, 16, 32]
    WINDOW_SIZE = 7
    MLP_RATIO = 4.
    QKV_BIAS = True
    QK_SCALE = None
    DROP_RATE = 0.2 # 0.01
    DROP_PATH_RATE = 0.2
    APE = False
    PATCH_NORM = True
    USE_CHECKPOINT = False
    PRETRAIN_CKPT = "pretrained_model/swin_tiny_patch4_window7_224.pth"
    NUM_CLASSES = 1

    # path
    use_dataaug = True
    tensorboardlog  = os.path.join("log", "tensorboard",name)
    train_data_path = 'datasets/CrackLS315_src_train.txt'
    val_data_path   = 'datasets/CrackLS315_src_val.txt'
    test_data_path  = 'datasets/CrackLS315_src_test.txt'
    checkpoint_path = './checkpoints'
    log_path = './log/'+'visdom/'+name+'.txt'
    saver_path = os.path.join(checkpoint_path, name)
    max_save = 30

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






'''
MODEL:
  TYPE: swin
  NAME: swin_base_patch4_window7_224_22kto1k_finetune
  DROP_PATH_RATE: 0.2
  SWIN:
    EMBED_DIM: 128
    DEPTHS: [ 2, 2, 18, 2 ]
    NUM_HEADS: [ 4, 8, 16, 32 ]
    WINDOW_SIZE: 7
TRAIN:
  EPOCHS: 30
  WARMUP_EPOCHS: 5
  WEIGHT_DECAY: 1e-8
  BASE_LR: 2e-05
  WARMUP_LR: 2e-08
  MIN_LR: 2e-07
  '''
if __name__ == "__main__":
    c = Config()
    c.show()