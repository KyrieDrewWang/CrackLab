import os
from pprint import pprint
from abc import ABC, abstractmethod
import setproctitle


class baseconfig(ABC):
    """This class is an abstract base class for config.
    Args:
        ABC (_type_): _description_
    """
    name:str           # exp name
    vis_env:str        # visdom visulizer name
    network_name:str   # model name

    gpu_id:str

    # scheduler
    scheduler:str
    warmup_steps:int
    warmup_learning_rate_multiplier:float # if 1: the target learning rate is the lr, elif > 1: the target learning rate is lr * warmup_learning_rate_multiplier, elif < 1: error.
    scheduler:str # "exp","step","mulitsteps"
    epoch:int
    pretrained_model=""
    weight_decay:float
    lr_decay:float
    lr_decay_step_size:int
    lr_decay_by_epoch:bool
    lr:float
    momentum:float
    use_adam:bool  # Use Adam optimizer
    betas:float # for Adam
    use_adamw:bool
    train_batch_size:int
    val_batch_size:int
    test_batch_size:int
    resize_width=None
    enlarge=False
    center_crop_size=None
    
    # loss function
    loss_mode:str  # "bce-rcf","bce","focalloss","diceloss"
    output_loss_weight:float
    fuse5_loss_weight:float
    fuse4_loss_weight:float
    fuse3_loss_weight:float
    fuse2_loss_weight:float
    fuse1_loss_weight:float
    acc_sigmoid_th:float
    pos_pixel_weight:float
    #focal loss
    GAMMA:float
    ALPHA:float
    
    # path
    use_dataaug = True
    train_data_path:str
    val_data_path:str
    checkpoint_path:str
    tensorboardlog:str
    log_path:str
    saver_path:str
    max_save:int

    # visdom
    port = 8091
    vis_train_loss_every = 500
    vis_train_acc_every  = 500
    vis_train_img_every  = 500
    vis_val_every        = 300

    # checkpointer
    save_format = ''
    save_ods = -1
    save_ois = -1
    save_acc = -1
    save_pos_acc = -1
    save_condition = 0.5 
    save_train_ods = -1
    save_train_ois = -1
    
    #args for swin model  
    use_convattn = True
    use_convproj = False
    use_DFN = True
    
    DE_use_convattn = True
    DE_use_convproj = False
    DE_use_DSF = True
    
    
    @abstractmethod
    def _parse(self, kwargs):
        pass

    @abstractmethod
    def _state_dict(self):
        pass
    @abstractmethod
    def show(self):
        pass
