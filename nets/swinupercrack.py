# coding=utf-8
from __future__ import absolute_import, division, print_function

import copy
import os
import sys
sys.path.append(".")
import math
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch
import torch.nn as nn

from config.config_swinupercrack import Config as cfg
from nets.swincrack_backbone import PatchEmbed, ConvPatchEmd, BasicLayer,PatchExpand, PatchMerging, FinalPatchExpand_X4
from nets.swincrack_skip_connetion import Fuse
from nets.swinupercrack_decoder import SwinUperBlock, FFN

class SwinCrackNet(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, 
                img_size=224, 
                patch_size=4, 
                in_chans=3, 
                num_classes=1000,
                embed_dim=96, 
                depths=[2, 2, 6, 2], 
                num_heads=[3, 6, 12, 24],
                window_size=7, 
                mlp_ratio=4., 
                qkv_bias=True, 
                qk_scale=None,
                drop_rate=0., 
                attn_drop_rate=0., 
                drop_path_rate=0.1,
                de_depths=[2, 2, 2, 2],
                de_num_heads=[32, 16, 8, 4],
                de_drop_rate=0.3,
                de_attn_drop_rate=0., 
                de_drop_path_rate=0.2,
                norm_layer=nn.LayerNorm, 
                ape=False, 
                patch_norm=True,
                use_checkpoint=False, **kwargs):
        super().__init__()

        print("SwinTransformerSys expand initial----depths:{};depths_decoder:{};drop_path_rate:{};num_classes:{}".format(
            depths,de_depths,drop_path_rate,num_classes))

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        
        if cfg.use_convEmbd:
            # Substitute mlp patch embed for conv patch embed
            self.patch_embed = ConvPatchEmd(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
                norm_layer=norm_layer if self.patch_norm else None,
                drop_rate=drop_rate
            )
        else:        
            self.patch_embed = PatchEmbed(
                patch_size=patch_size, 
                in_chans=in_chans, 
                embed_dim=embed_dim,
                norm_layer=norm_layer if self.patch_norm else None)
        

        
        num_patches = math.ceil(img_size/patch_size)
        patches_resolution = (math.ceil(img_size/patch_size), math.ceil(img_size/patch_size))
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        de_dpr = [x.item() for x in torch.linspace(0, de_drop_path_rate, sum(de_depths))]
        
        # build encoder and bottleneck layers
        self.layers_down = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer_down = BasicLayer(
                                dim=int(embed_dim * 2 ** i_layer),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias,
                                qk_scale=qk_scale,
                                drop=drop_rate, 
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint)
            self.layers_down.append(layer_down)
        
        #build skip connection
        self.ffns = nn.ModuleList()
        for i in range(0, self.num_layers-1):
            mlp = FFN(
                dims=embed_dim*2**i,
                mlp_ratio=mlp_ratio,
                act_layer=nn.GELU,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate)
            self.ffns.append(mlp)
        
        
        # build decoder layers
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            
            if cfg.use_SC:
                # substitute linear concat for conv concat back in SC
                concat_linear = nn.Sequential(
                    Rearrange('b (h w) c -> b c h w', 
                            h=patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)), 
                            w=patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    nn.Conv2d(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                            int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                            kernel_size=3, 
                            stride=1, 
                            padding=1), 
                    nn.GELU(),
                    nn.Conv2d(int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                            int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                            kernel_size=3, 
                            stride=1, 
                            padding=1), 
                    nn.GELU(),
                    nn.Dropout(p=drop_rate),
                    Rearrange('b c h w -> b (h w) c', 
                            h=patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),w=patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))))
            else:
                concat_linear = nn.Linear(
                    2*int(embed_dim*2**(self.num_layers-1-i_layer)),
                    int(embed_dim*2**(self.num_layers-1-i_layer))) if i_layer > 0 else nn.Identity()
        
            if i_layer ==0 :
                layer_up = PatchExpand(dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer)), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = SwinUperBlock(
                        embed_dims=int(embed_dim * 2 ** (self.num_layers-1-i_layer)),
                        num_heads=de_num_heads[i_layer],
                        depth=de_depths[i_layer],
                        window_size=window_size,
                        qkv_bias=qkv_bias, 
                        qk_scale=qk_scale,
                        drop_rate=de_drop_rate, 
                        attn_drop_rate=de_attn_drop_rate,
                        drop_path_rate=de_dpr[sum(de_depths[:(i_layer)]):sum(de_depths[:(i_layer) + 1])],
                        mlp_ratio = self.mlp_ratio, 
                        norm_layer=norm_layer,
                        act_layer=nn.GELU,
                        is_upsample=True if i_layer < self.num_layers - 1 else False)
                
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm_down = norm_layer(self.num_features)
        self.norm_up= norm_layer(2*self.embed_dim)
        
        self.up = FinalPatchExpand_X4(input_resolution=self.patches_resolution,dim_scale=4,dim=2*embed_dim)
        self.output = nn.Conv2d(in_channels=2*embed_dim,out_channels=self.num_classes,kernel_size=1,bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    #Encoder and Bottleneck
    def forward_down_features(self, x):
        x, H, W = self.patch_embed(x)   # torch.Size([1, 16384, 128])
        
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        x_downsample = []  # store the output of the residual connection of each stage

        for layer_down in self.layers_down:
            x_downsample.append(x)  # todo: the residual connection from where ??
            x, H, W, _ = layer_down(x, H, W)

        x = self.norm_down(x)  # B L C
  
        return x, x_downsample, H, W

    #Decoder and Skip connection
    def forward_up_features(self, x, x_downsample, H, W):
        up_wh=(H, W)
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x, H, W = layer_up(x, H, W)
                up_wh=(H, W)
            else:
                x = layer_up(x, x_downsample[3-inx], up_wh)
                if layer_up.is_upsample:
                    up_wh=(up_wh[0]*2, up_wh[1]*2)
        x = self.norm_up(x)  # B L C
  
        return x, up_wh[0], up_wh[1]

    def up_x4(self, x, H, W):
        B, L, C = x.shape
        assert L == H*W, "input features has wrong size"

        x = self.up(x)
        x = x.view(B,4*H,4*W,-1)
        x = x.permute(0,3,1,2) #B,C,H,W
        x = self.output(x)
            
        return x

    def forward(self, x):
        x, x_downsample, H, W = self.forward_down_features(x)
        for inx, ffn in enumerate(self.ffns):
            skip_x = ffn(x_downsample[inx])
            x_downsample[inx] = skip_x
        x, H, W = self.forward_up_features(x,x_downsample, H, W)
        x = self.up_x4(x, H, W)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers_down):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops



class swinupercrack(nn.Module):
    def __init__(self, config=cfg):
        super(swinupercrack, self).__init__()
        self.config = config

        self.swin_unet = SwinCrackNet(
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
                                attn_drop_rate=config.ATTN_DROP_RATE,
                                ape=config.APE,
                                patch_norm=config.PATCH_NORM,
                                use_checkpoint=config.USE_CHECKPOINT,
                                de_depths=config.DE_DEPTHS,
                                de_num_heads=config.DE_NUM_HEADS,
                                de_drop_rate=config.DE_DROP_RATE,
                                de_attn_drop_rate=config.DE_ATTN_DROP_RATE, 
                                de_drop_path_rate=config.DE_DROP_PATH_RATE)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.swin_unet(x)
        return logits, logits, logits, logits, logits, logits

    def load_from(self, config=cfg):
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
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id
    inp = torch.randn(1, 3, 512, 512)
    net = swinupercrack()
    net.load_from(cfg)
    out=net(inp)
    print(out[-1].shape)