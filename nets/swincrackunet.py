# coding=utf-8
from __future__ import absolute_import, division, print_function

import copy
import sys

import torch
import torch.nn as nn

sys.path.append(".")
from nets.swincrackunet_network import SwinCrackUnet

sys.path.append(".")
from tensorboardX import SummaryWriter
from thop import profile

from config.config_swincrackunet import Config as cfg


class SwinUnet(nn.Module):
    def __init__(self, config, num_classes=21843, zero_head=False):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config

        self.swin_unet = SwinCrackUnet(
                                img_size=config.IMG_SIZE,
                                patch_size=config.PATCH_SIZE,
                                in_chans=config.IN_CHANS,
                                num_classes=config.NUM_CLASSES,
                                embed_dim=config.EMBED_DIM,
                                depths=config.DEPTHS,
                                num_heads=config.NUM_HEADS,
                                window_size=config.WINDOW_SIZE,
                                mlp_ratio=config.MLP_RATIO,
                                qkv_bias=config.QKV_BIAS,
                                qk_scale=config.QK_SCALE,
                                drop_rate=config.DROP_RATE,
                                drop_path_rate=config.DROP_PATH_RATE,
                                ape=config.APE,
                                patch_norm=config.PATCH_NORM,
                                use_checkpoint=config.USE_CHECKPOINT)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits, buf = self.swin_unet(x)
        return logits.squeeze(1)

    def load_from(self, config):
        pretrained_path = config.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    if int(k[7:8]) == 3:
                        down_k = "layers_down." + str(int(k[7:8]) + 1) + k[8:]
                        full_dict.update({down_k:v})
                        up_num = len(config.DEPTHS) - int(k[7:8]) - 2
                        up_k = "layers_up." + str(up_num) + k[8:]
                        full_dict.update({up_k:v})
                        del full_dict[k]
                        continue
                    if int(k[7:8]) == 2:
                        down_k = "layers_down." + str(int(k[7:8])+1) + k[8:]
                        full_dict.update({down_k:v})
                        up_k = "layers_up." + str(len(config.DEPTHS) - int(k[7:8]) - 2) + k[8:]
                        full_dict.update({up_k:v})
                    down_k = "layers_down." + k[7:]
                    full_dict.update({down_k:v})
                    up_num = len(config.DEPTHS) - int(k[7:8]) - 1
                    up_k = "layers_up." + str(up_num) + k[8:]
                    full_dict.update({up_k:v})
                    del full_dict[k]

            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")


if __name__ == '__main__':
    inp = torch.randn(1, 3, 512, 512)
    net = SwinUnet(cfg)
    net.load_from(cfg)
    #with SummaryWriter(log_dir="model") as sw:
        #sw.add_graph(net, inp)
    #flops, params = profile(net, inputs=(inp, ))
    #print(flops, params)
    out=net(inp)
    print(out.shape)
   