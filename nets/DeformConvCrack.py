import sys
sys.path.append('.')
from config.config_DeformConvCrack import Config as cfg
import os
os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_id
import copy
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import trunc_normal_, DropPath
from ops_dcnv3 import modules as opsm
import torch.nn.functional as F
from ptflops import get_model_complexity_info
from nets.crackformer import Trans_EB

class to_channels_first(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class to_channels_last(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)


def build_norm_layer(dim,
                     norm_layer,
                     in_format='channels_last',
                     out_format='channels_last',
                     eps=1e-6):
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(
            f'build_norm_layer does not support {norm_layer}')
    return nn.Sequential(*layers)


def build_act_layer(act_layer):
    if act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_layer == 'GELU':
        return nn.GELU()

    raise NotImplementedError(f'build_act_layer does not support {act_layer}')

class PA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pa_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.pa_conv(x))

class StemLayer(nn.Module):
    r""" Stem layer of InternImage
    Args:
        in_chans (int): number of input channels
        out_chans (int): number of output channels
        act_layer (str): activation layer
        norm_layer (str): normalization layer
    """

    def __init__(self,
                 in_chans=3,
                 out_chans=96,
                 act_layer='GELU',
                 norm_layer='BN'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans,
                               out_chans // 2,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.norm1 = build_norm_layer(out_chans // 2, norm_layer,
                                      'channels_first', 'channels_first')
        self.act = build_act_layer(act_layer)
        self.conv2 = nn.Conv2d(out_chans // 2,
                               out_chans,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        if cfg.PA:
            self.pa = PA(out_chans)
        
        self.norm2 = build_norm_layer(out_chans, norm_layer, 'channels_first', 'channels_last')

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        if cfg.PA:
            x = self.pa(x)
        x = self.norm2(x)
        return x
    

class Stempro(nn.Module):
    def __init__(self,
                in_chans=3,
                out_chans=96,
                act_layer='GELU',
                norm_layer='BN',
                with_pos=False) -> None:
        super().__init__()
        self.with_pos = with_pos
        if self.with_pos:
            self.pa = PA(out_chans)
        self.conv1 = nn.Conv2d(in_channels=in_chans,
                                out_channels=out_chans//2,
                                kernel_size=3,
                                stride=2,
                                padding=1)
        self.norm1 = build_norm_layer(out_chans//2, norm_layer, 'channels_first', 'channels_first')
        self.act1 = build_act_layer(act_layer)
        self.conv2 = nn.Conv2d(out_chans//2,
                               out_chans//2,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.norm2 = build_norm_layer(out_chans//2,
                                      norm_layer,
                                      'channels_first',
                                      'channels_first')
        self.act2 = build_act_layer(act_layer)
        self.conv3 = nn.Conv2d(out_chans//2, out_chans, 3, 2, 1)
        self.norm3 = build_norm_layer(out_chans, norm_layer, 'channels_first', 'channels_last')
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)

        x = self.conv3(x)
        if self.with_pos:
            x = self.pa(x)
        x = self.norm3(x)
        return x
    

class DownsampleLayer(nn.Module):
    r""" Downsample layer of InternImage
    Args:
        channels (int): number of input channels
        norm_layer (str): normalization layer
    """

    def __init__(self, channels, norm_layer='BN'):
        super().__init__()
        
        if cfg.SE_DOWN:
            self.norm_se = build_norm_layer(channels, norm_layer, 'channels_last', 'channels_first')
            self.depth_conv = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
            self.act = nn.GELU()
            self.ca = ChannelAttention(channels)
            self.sa = SpatialAttention(kernel_size=7)
            self.conv_se = nn.Conv2d(channels, channels, 1, 1, bias=False)
        self.conv = nn.Conv2d(channels,
                              2 * channels,
                              kernel_size=3,
                              stride=2,
                              padding=1,
                              bias=False)
        # self.maxpool = torch.nn.MaxPool2d(kernel_size=2,stride=2, return_indices=True)
        self.norm = build_norm_layer(2 * channels, norm_layer, 'channels_first', 'channels_last')

    def forward(self, x):
        if cfg.SE_DOWN:
            x_ = self.norm_se(x)
            x_ = self.depth_conv(x_)
            x_ = self.act(x_)
            x_ = self.ca(x_) * x_
            x_ = self.sa(x_) * x_
            x_ = self.conv_se(x_)
            x = x_ + x.permute(0, 3, 1, 2)
            x = x.permute(0, 2, 3, 1)
        x = self.conv(x.permute(0, 3, 1, 2))
        # x, indices = self.maxpool(x)
        x = self.norm(x)
        
        return x

class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B,H,W,C = x.shape
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        x = self.norm(x)
        x = self.reduction(x) # [B, H/2*W/2, 2*C]
        x = x.view(B, H//2, W//2, C*2)
        return x

# class PoolLayer(nn.Module):
    
#     def __init__(self) -> None:
#         super().__init__()
        
#     def forward(self, ):
#         pass
    
    
class UpsampleLayer(nn.Module):
    def __init__(self, channels, scale=2, norm_layer='LN', act_layer='GELU'):
        super().__init__()
        self.scale = scale
        # self.unpool = torch.nn.MaxUnpool2d(2, 2)
        self.conv1 = nn.Conv2d(
            channels,
            channels//2,
            kernel_size=3,
            stride=1,
            padding=1)
        self.norm1 = build_norm_layer(channels//2, norm_layer, 'channels_first', 'channels_first')
        self.act = build_act_layer(act_layer)
        # self.conv2 = nn.Conv2d(channels//2,
        #                        channels//2,
        #                        kernel_size=3,
        #                        stride=1,
        #                        padding=1)
        # self.norm2 = build_norm_layer(channels//2, norm_layer, 'channels_first', 'channels_last')
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = F.interpolate(x, scale_factor=self.scale, mode='bilinear')
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        # x = self.conv2(x)
        # x = self.norm2(x)
        return x.permute(0, 2, 3, 1)

class PatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.ps = nn.PixelShuffle(2)
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Linear(dim, dim, bias=False)
        self.norm = norm_layer(dim // 4 if dim_scale==1 else dim//2)
    def forward(self, x):
        B, H, W, C  = x.shape
        x = self.expand(x)
        x = x.view(B, H, W, -1)
        x = x.permute(0,3,1,2)  # B, C, H, W
        x = self.ps(x)
        x = x.permute(0, 2, 3, 1)  
        x= self.norm(x)
        return x


class FinalUpsample(nn.Module):
    
    def __init__(self, 
                 in_channels,
                 num_class, 
                 act_layer='GELU',
                 norm_layer='BN') -> None:
        super().__init__()
        self.num_class = num_class
        self.conv1 = nn.Conv2d(in_channels, in_channels//2, 3, 1, 1)
        self.norm1 = build_norm_layer(in_channels // 2, norm_layer,
                                      'channels_first', 'channels_first')
        self.act1 = build_act_layer(act_layer)
        self.conv2 = nn.Conv2d(in_channels//2, num_class, 3, 1, 1)
        # self.conv3 = nn.Conv2d(in_channels//2, num_class, 3, 1, 1)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=4, mode="bilinear")
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        return x.permute(0, 2, 3, 1)


class MLPLayer(nn.Module):
    r""" MLP layer of InternImage
    Args:
        in_features (int): number of input features
        hidden_features (int): number of hidden features
        out_features (int): number of output features
        act_layer (str): activation layer
        drop (float): dropout rate
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer='GELU',
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = build_act_layer(act_layer)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class StageOut(nn.Module):  
    def __init__(self, channels, embed_dims, scale) -> None:
        super().__init__()
        self.channels = channels
        self.embed_dims = embed_dims
        self.scale = scale
        self.conv = nn.Conv2d(channels, channels, 1, 1)
        self.conv1 = nn.Conv2d(channels, self.embed_dims, kernel_size=3, padding=1, stride=1)
        self.norm1 = build_norm_layer(self.embed_dims, 'BN', 'channels_first', 'channels_first')
        self.act1  = nn.GELU()
        self.conv2 = nn.Conv2d(self.embed_dims, self.embed_dims//2, kernel_size=3, padding=1, stride=1)
        self.norm2 = build_norm_layer(self.embed_dims//2, 'BN', 'channels_first', 'channels_first')
        self.act2  = nn.GELU()
        self.conv3 = nn.Conv2d(self.embed_dims//2, 1, 1, 1)
    def forward(self, x):
        """
        input:  [B, H, W, C]
        output: [B, C, H, W]
        """
        x = x.permute(0,3,1,2)
        out = self.conv(x)
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.act1(out)
        out = F.interpolate(out, scale_factor=self.scale, mode='bilinear')
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)
        out = self.conv3(out)
        return out

class SideOut(nn.Module):  
    def __init__(self, channels, scale, embed_dims, norm_layer='LN') -> None:
        super().__init__()
        self.scale = scale
        self.conv = nn.Conv2d(channels, channels//2, 1)
        # self.conv1 = nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=3, padding=1, stride=1)
        # self.norm1 = build_norm_layer(self.embed_dims, norm_layer, 'channels_first', 'channels_first')
        # self.act1 = nn.GELU()
    def forward(self, x):
        """
        input:  [B, H, W, C]
        output: [B, C, H, W]
        """
        x = x.permute(0,3,1,2)
        x = self.conv(x)
        out = F.interpolate(x, scale_factor=self.scale, mode='bilinear')
        # out = self.conv1(out)
        # out = self.norm1(out)
        # out = self.act1(out)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(channels, channels // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(channels // 16, channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(2, 2, 7, 1, 3, groups=2, bias=False),
        #     nn.Conv2d(2, 1, 1, bias=False)
        # )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ConvBlock(nn.Module):
    def __init__(self, channles) -> None:
        super().__init__()
        self.channles = channles
        self.conv1 = nn.Conv2d(channles, channles, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(channles)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(channles, channles, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(channles)
        self.act2 = nn.GELU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x


class CBAMup(nn.Module):  
    def __init__(self, channels, scale, embdchanls, num_class=1, norm_layer='BN') -> None:
        super().__init__()
        self.channels = channels
        self.scale = scale
        self.embdchanls = embdchanls
        self.ca = ChannelAttention(self.channels)
        self.sa = SpatialAttention(kernel_size=7)
        self.conv1 = nn.Conv2d(self.channels, self.embdchanls, 3, 1, 1)
        self.norm1 = build_norm_layer(self.embdchanls, norm_layer, 'channels_first', 'channels_first')
        self.act1 = build_act_layer('GELU')
        self.conv2 = nn.Conv2d(self.embdchanls, self.embdchanls//2, 3, 1, 1)
        self.norm2 = build_norm_layer(self.embdchanls//2, norm_layer, 'channels_first', 'channels_first')
        self.act2 = build_act_layer('GELU')
    def forward(self, x):
        """
        input:  [B, H, W, C]
        output: [B, C, H, W]
        """
        # CBAM:
        x = x.permute(0,3,1,2)
        x = self.ca(x) * x
        x = self.sa(x) * x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
              
        # Upsample:
        out = F.interpolate(out, scale_factor=self.scale, mode='bilinear')
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)
        return out


class DSF(nn.Module):
    def __init__(self, 
                 dim, 
                 drop_path=0.2,
                 kernel_size=3,
                 layer_scale_init_value=0.9):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)  # depthwise conv 7,3  5,2  3,1
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)  # nn.Linear(4 * dim, dim)
        # self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        # x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # (N, C, H, W)
        x = self.pwconv1(x)        # (N, C, H, W)
        x = self.act(x)
        x = self.pwconv2(x)        # (N, C, H, W)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.drop_path(x)
        x = x.permute(0, 3, 1, 2)
        x = input + x
        return x
    
    
class UpCBAM(nn.Module):  
    def __init__(self, channels, embdchans, norm_layer='BN') -> None:
        super().__init__()
        self.channels = channels
        self.ca = ChannelAttention(self.channels)
        self.sa = SpatialAttention(kernel_size=7)
        # self.convblock = ConvBlock(self.channels)
        # self.dsf = DSF(self.channels, kernel_size=3)
        self.conv1 = nn.Conv2d(self.channels, embdchans, 3, 1, 1)
        self.norm1 = build_norm_layer(embdchans, norm_layer, 'channels_first', 'channels_first')
        self.act1 = build_act_layer('GELU')
        self.conv2 = nn.Conv2d(embdchans, embdchans//2, 3, 1, 1)
        self.norm2 = build_norm_layer(embdchans//2, norm_layer, 'channels_first', 'channels_first')
        self.act2 = build_act_layer('GELU')
    def forward(self, x):
        """
        input:  [B, H, W, C]
        output: [B, C, H, W]
        """
        # CBAM:
        # x = x.permute(0,3,1,2)
        x = self.ca(x) * x
        x = self.sa(x) * x
        # x = self.dsf(x)
        # x = self.convblock(x)
        
        # output:
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = F.interpolate(x, scale_factor=4, mode="bilinear")
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)

        return x

class CBAMdecoder(nn.Module):
    def __init__(self, channels, upsample=False, start=False) -> None:
        super().__init__()
        self.upsample = upsample
        self.start = start
        self.conv1 = nn.Conv2d(channels, channels, 1) if start else None
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention(kernel_size=7)
        self.convblock = ConvBlock(channels)
        self.upsample = nn.Sequential(
            nn.Conv2d(channels, channels//2, 3, 1, 1),
            nn.BatchNorm2d(channels//2),
            nn.ReLU()
        ) if upsample else None
    def forward(self, x, sideout=False):
        x = x.permute(0, 3, 1, 2)
        if self.start:
            x = self.conv1(x)
        x = self.ca(x) * x
        x = self.sa(x) * x
        x = self.convblock(x)
        if sideout:
            _x = x
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear')
            x = self.upsample(x) 
        if sideout:
            return x.permute(0, 2, 3, 1), _x.permute(0, 2, 3, 1)
        return x.permute(0, 2, 3, 1)
    
    
class CBAMagg(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.channels = channels
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()
        self.norm = build_norm_layer(channels, 'LN', 'channels_first', 'channels_last')
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.ca(x)*x
        x = self.sa(x)*x
        x = self.norm(x)
        return x
        

class AttentionGate(nn.Module):
    def __init__(self, input_channels, output_channels, act_layer='GELU', norm_layer='BN'):
        super(AttentionGate, self).__init__()
        self.sigma = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(output_channels//4,output_channels)
        )
        
        self.fi = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(output_channels//4,output_channels),
            nn.Sigmoid()
        )
        self.act1 =build_act_layer(act_layer)
        self.conv1=nn.Conv2d(2*input_channels, input_channels, 1, 1)
        self.act2 = build_act_layer(act_layer)
        self.norm1 = build_norm_layer(input_channels, norm_layer, 'channels_first', 'channels_last')
    def forward(self, skip, up):
        skip = skip.permute(0, 3, 1, 2)
        up = up.permute(0, 3, 1, 2)        
        sum = skip+up
        sum = self.act1(sum)
        out = self.sigma(sum)
        att = self.fi(out)  # Mask  #B, C, H, W
        
        skip_attn = att*skip
        outputs = torch.cat([skip_attn, up], 1) 
        outputs = self.conv1(outputs)
        outputs = self.act2(outputs)
        outputs = self.norm1(outputs)
        return outputs  # .permute(0, 2, 3, 1)

'''
class AttentionGate(nn.Module):
    # AG pro 1
    def __init__(self, input_channels, output_channels):
        super(AttentionGate, self).__init__()
        self.sigma = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(output_channels//4,output_channels)
        )
        
        self.fi = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(output_channels//4,output_channels),
            nn.Sigmoid()
        )
        # self.relu = nn.ReLU(inplace=True)
        self.gelu=nn.GELU()
        self.conv1=nn.Conv2d(2*input_channels, input_channels, 3, 1, 1)
        self.act1 = nn.ReLU()
        self.conv2=nn.Conv2d(input_channels, input_channels, 3, 1, 1)
    def forward(self, skip, up):
        skip = skip.permute(0, 3, 1, 2)
        up = up.permute(0, 3, 1, 2)
        outputs = torch.cat([skip, up], 1)
        
        sum = skip+up
        sum = self.gelu(sum)
        out = self.sigma(sum)
        att = self.fi(out)  # Mask  #B, C, H, W
        
        outputs = self.conv1(outputs)
        outputs = self.act1(outputs)
        outputs = att*outputs
        outpsut  = self.conv2(outputs)
        return outpsut.permute(0, 2, 3, 1)
'''


class AttentionGateD(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.channels = channels
        self.q = nn.Conv2d(self.channels, self.channels, 1)
        self.bn_q = nn.BatchNorm2d(self.channels)#build_norm_layer(self.channels, 'BN', 'channels_first', 'channels_first')
        self.v    = nn.Conv2d(self.channels, self.channels, 1)
        self.bn_v = nn.BatchNorm2d(self.channels)#build_norm_layer(self.channels, 'BN', 'channels_first', 'channels_first')
        self.RELU = nn.ReLU()
        self.conv = nn.Conv2d(self.channels, self.channels, 1)
        self.bn   = nn.BatchNorm2d(self.channels)#build_norm_layer(self.channels, 'BN', 'channels_first', 'channels_first')
        self.sigmoid = nn.Sigmoid()
    def forward(self, q, v):
        q = self.q(q.permute(0, 3, 1, 2))
        q = self.bn_q(q)
        v = self.v(v.permute(0, 3, 1, 2))
        v = self.bn_v(v)
        qv = q + v
        qv = self.RELU(qv)
        qv = self.conv(qv)
        qv = self.bn(qv)
        qv_attn = self.sigmoid(qv)
        return qv_attn.permute(0, 2, 3 ,1)
    
'''
class AttentionGateD(nn.Module):
    # AGD pro 1
    def __init__(self, channels) -> None:
        super().__init__()
        self.channels = channels
        self.q = nn.Conv2d(self.channels, self.channels, 3, 1, 1)
        self.bn_q = nn.BatchNorm2d(self.channels)#build_norm_layer(self.channels, 'BN', 'channels_first', 'channels_first')
        self.v    = nn.Conv2d(self.channels, self.channels, 3, 1, 1)
        self.bn_v = nn.BatchNorm2d(self.channels)#build_norm_layer(self.channels, 'BN', 'channels_first', 'channels_first')
        self.GELU = nn.GELU()
        self.conv = nn.Conv2d(self.channels, self.channels, 3, 1, 1)
        self.bn   = nn.BatchNorm2d(self.channels)#build_norm_layer(self.channels, 'BN', 'channels_first', 'channels_first')
        self.sigmoid = nn.Sigmoid()
    def forward(self, q, v):
        q = self.q(q.permute(0, 3, 1, 2))
        q = self.bn_q(q)
        v = self.v(v.permute(0, 3, 1, 2))
        v = self.bn_v(v)
        qv = q + v
        qv = self.GELU(qv)
        qv = self.conv(qv)
        qv = self.bn(qv)
        qv_attn = self.sigmoid(qv)
        return qv_attn.permute(0, 2, 3 ,1)
'''


class Rectification(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)
    def forward(self, x):
        x = self.conv(x)
        attn = torch.sigmoid(x)
        return x, attn

class LocalEnhance(nn.Module):
    def __init__(self,channels, norm_layer='LN') -> None:
        super().__init__()
        self.conv1_1 = nn.Sequential(nn.Conv2d(channels, channels, 1, 1), 
                                     build_norm_layer(channels, norm_layer, 'channels_first', 'channels_first'),
                                     build_act_layer('GELU'))
        
        self.conv1_2 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1), 
                                     build_norm_layer(channels, norm_layer, 'channels_first', 'channels_first'), 
                                     build_act_layer('GELU'))
        
        self.conv2_1 = nn.Sequential(nn.Conv2d(channels, channels, 1, 1), 
                                     build_norm_layer(channels, norm_layer, 'channels_first', 'channels_first'), 
                                     build_act_layer('GELU'))
        
        self.conv2_2 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1), 
                                     build_norm_layer(channels, norm_layer, 'channels_first', 'channels_first'), 
                                     build_act_layer('GELU'))
        
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x1_1 = self.conv1_1(x)
        x1_2 = self.conv1_2(x)

        x = x1_1 + x1_2
        
        x2_1 = self.conv2_1(x)
        x2_2 = self.conv2_2(x)

        x = x2_1 + x2_2
        return x.permute(0, 2, 3, 1)
    
'''    
class LocalEnhanceFFN(nn.Module):
    def __init__(self, channels, hidden=4) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels*hidden, kernel_size=1, stride=1)
        self.act1 = nn.GELU()
        self.depthconv = nn.Conv2d(in_channels=channels*hidden, out_channels=channels*hidden, kernel_size=7, padding=3, stride=1,groups=channels*hidden)
        self.act2 = nn.GELU()
        self.conv2 = nn.Conv2d(in_channels=channels*hidden, out_channels=channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.depthconv(x)
        x = self.act2(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1)
        return x
'''

class LocalEnhanceFFN(nn.Module):
    
    def __init__(self, channels, hidden=4, LayerNorm='LN', layer_scale_init_value=0.7, drop_path=0.4) -> None:
        super().__init__()
        self.depthconv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=7, padding=3, stride=1,groups=channels)
        self.norm = build_norm_layer(channels, LayerNorm, 'channels_first', 'channels_first')
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels*hidden, kernel_size=1, stride=1)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(in_channels=channels*hidden, out_channels=channels, kernel_size=1, stride=1)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((channels)),requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, x):
        input = x
        x = x.permute(0, 3, 1, 2)
        x = self.depthconv(x)
        x = self.norm(x)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1)
        if self.gamma is not None:
            x = self.gamma * x
        x = input + self.drop_path(x)  # (N, H, W, C)
        return x

class InternImageLayer(nn.Module):
    r""" Basic layer of InternImage
    Args:
        core_op (nn.Module): core operation of InternImage
        channels (int): number of input channels
        groups (list): Groups of each block.
        mlp_ratio (float): ratio of mlp hidden features to input channels
        drop (float): dropout rate
        drop_path (float): drop path rate
        act_layer (str): activation layer
        n,orm_layer (str): normalization layer
        post_norm (bool): whether to use post normalization
        layer_scale (float): layer scale
        offset_scale (float): offset scale
        with_cp (bool): whether to use checkpoint
    """

    def __init__(self,
                 core_op,
                 channels,
                 groups,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer='GELU',
                 norm_layer='LN',
                 post_norm=False,
                 layer_scale=None,
                 offset_scale=1.0,
                 with_cp=False):
        super().__init__()
        self.channels = channels
        self.groups = groups
        self.mlp_ratio = mlp_ratio
        self.with_cp = with_cp

        self.norm1 = build_norm_layer(channels, 'LN')
        self.post_norm = post_norm
        self.dcn = core_op(channels=channels,
                           kernel_size=3,
                           stride=1,
                           pad=1,
                           dilation=1,
                           group=groups,
                           offset_scale=offset_scale,
                           act_layer=act_layer,
                           norm_layer=norm_layer)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = build_norm_layer(channels, 'LN')
        
        if cfg.LOCAL_ENHANCE_FFN:
            self.mlp = LocalEnhanceFFN(channels=channels)
        else:
            self.mlp = MLPLayer(in_features=channels,
                            hidden_features=int(channels * mlp_ratio),
                            act_layer=act_layer,
                            drop=drop)
        
        # self.dsf = DSF(dim=channels,
        #                drop_path=drop_path,
        #                kernel_size=7)
        self.layer_scale = layer_scale is not None
        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(channels),
                                       requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(channels),
                                       requires_grad=True)

    def forward(self, x):

        def _inner_forward(x):
            if not self.layer_scale:
                if self.post_norm:
                    x = x + self.drop_path(self.norm1(self.dcn(x)))
                    x = x + self.drop_path(self.norm2(self.mlp(x)))
                else:
                    x = x + self.drop_path(self.dcn(self.norm1(x)))
                    x = x + self.drop_path(self.mlp(self.norm2(x)))
                return x
            if self.post_norm:
                x = x + self.drop_path(self.gamma1 * self.norm1(self.dcn(x)))
                x = x + self.drop_path(self.gamma2 * self.norm2(self.mlp(x)))
                # x = x + self.drop_path(self.gamma1 * self.norm1(self.dcn(x)))
                # x = x + self.drop_path(self.gamma2 * self.norm2(self.dsf(x)))
            else:
                x = x + self.drop_path(self.gamma1 * self.dcn(self.norm1(x)))
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
            return x

        if self.with_cp and x.requires_grad:
            x = checkpoint.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class InternImageBlock(nn.Module):
    r""" Block of InternImage
    Args:
        core_op (nn.Module): core operation of InternImage
        channels (int): number of input channels
        depths (list): Depth of each block.
        groups (list): Groups of each block.
        mlp_ratio (float): ratio of mlp hidden features to input channels
        drop (float): dropout rate
        drop_path (float): drop path rate
        act_layer (str): activation layer
        norm_layer (str): normalization layer
        post_norm (bool): whether to use post normalization
        layer_scale (float): layer scale
        offset_scale (float): offset scale
        with_cp (bool): whether to use checkpoint
    """

    def __init__(self,
                 core_op,
                 channels,
                 depth,
                 groups,
                 downsample=True,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer='GELU',
                 norm_layer='LN',
                 post_norm=False,
                 offset_scale=1.0,
                 layer_scale=None,
                 with_cp=False):
        super().__init__()
        self.channels = channels
        self.depth = depth
        self.post_norm = post_norm

        self.blocks = nn.ModuleList([
            InternImageLayer(core_op=core_op,
                             channels=channels,
                             groups=groups,
                             mlp_ratio=mlp_ratio,
                             drop=drop,
                             drop_path=drop_path[i] if isinstance(
                                 drop_path, list) else drop_path,
                             act_layer=act_layer,
                             norm_layer=norm_layer,
                             post_norm=post_norm,
                             layer_scale=layer_scale,
                             offset_scale=offset_scale,
                             with_cp=with_cp) for i in range(depth)
        ])
        if not self.post_norm:
            self.norm = build_norm_layer(channels, 'LN')
        
        if cfg.PIXEL_SHUFFLE_DOWN:
            self.downsample = PatchMerging(channels) if downsample else None
        else:
            self.downsample = DownsampleLayer(
            channels=channels, norm_layer=norm_layer) if downsample else None
        
        
    def forward(self, x, return_wo_downsample=False):
        for blk in self.blocks:
            x = blk(x)
        if not self.post_norm:
            x = self.norm(x)
        if return_wo_downsample:
            x_ = x

            
        if self.downsample is not None:
            x = self.downsample(x)
        if return_wo_downsample:
            return x, x_
        return x

class LambdUp(nn.Module):
    def __init__(self, channels, num_layer=2, upsample=True) -> None:
        super().__init__()
        self.lambd1 = Trans_EB(channels, channels)
        self.lambd2 = Trans_EB(channels, channels)
        self.lambd3 = Trans_EB(channels, channels) if num_layer == 3 else None
        self.upsample = UpsampleLayer(channels) if upsample else None
        
    def forward(self, x, sideout=False):
        x = x.permute(0, 3, 1, 2)
        x = self.lambd1(x)
        x = self.lambd2(x)
        x = self.lambd3(x)
        if sideout:
            x_ = x
        if self.upsample:
            x = self.upsample(x)
        if sideout:
            return x, x_
        return x


class InternImageBlock_up(nn.Module):
    r""" Block of InternImage
    Args:
        core_op (nn.Module): core operation of InternImage
        channels (int): number of input channels
        depths (list): Depth of each block.
        groups (list): Groups of each block.
        mlp_ratio (float): ratio of mlp hidden features to input channels
        drop (float): dropout rate
        drop_path (float): drop path rate
        act_layer (str): activation layer
        norm_layer (str): normalization layer
        post_norm (bool): whether to use post normalization
        layer_scale (float): layer scale
        offset_scale (float): offset scale
        with_cp (bool): whether to use checkpoint
    """

    def __init__(self,
                 core_op,
                 channels,
                 depth,
                 groups,
                 upsample=True,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer='GELU',
                 norm_layer='LN',
                 post_norm=False,
                 offset_scale=1.0,
                 layer_scale=None,
                 with_cp=False):
        super().__init__()
        self.channels = channels
        self.depth = depth
        self.post_norm = post_norm
        if cfg.CBAMAGG:
            self.cbamagg = CBAMagg(channels=channels)
        self.blocks = nn.ModuleList([
            InternImageLayer(core_op=core_op,
                             channels=channels,
                             groups=groups,
                             mlp_ratio=mlp_ratio,
                             drop=drop,
                             drop_path=drop_path[i] if isinstance(
                                 drop_path, list) else drop_path,
                             act_layer=act_layer,
                             norm_layer=norm_layer,
                             post_norm=post_norm,
                             layer_scale=layer_scale,
                             offset_scale=offset_scale,
                             with_cp=with_cp) for i in range(depth)
        ])
        if not self.post_norm:
            self.norm = build_norm_layer(channels, 'LN')
        
        if cfg.PIXEL_SHUFFLE_UP:
            self.upsample = PatchExpand(channels) if upsample else None
            
        else:
            self.upsample = UpsampleLayer(
                channels=channels, 
                norm_layer=norm_layer) if upsample else None

    def forward(self, 
                x, 
                sideout=False):
        for blk in self.blocks:
            x = blk(x)
        if not self.post_norm:
            x = self.norm(x)
        if sideout:
            x_ = x
        if cfg.CBAMAGG:
            x = self.cbamagg(x)
        if self.upsample is not None:
            x = self.upsample(x)
        if sideout:
            return x, x_
        return x



class InternImage(nn.Module):
    r""" InternImage
        A PyTorch impl of : `InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        core_op (str): Core operator. Default: 'DCNv3'
        channels (int): Number of the first stage. Default: 64
        depths (list): Depth of each block. Default: [3, 4, 18, 5]
        groups (list): Groups of each block. Default: [3, 6, 12, 24]
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        drop_rate (float): Probability of an element to be zeroed. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        act_layer (str): Activation layer. Default: 'GELU'
        norm_layer (str): Normalization layer. Default: 'LN'
        layer_scale (bool): Whether to use layer scale. Default: False
        cls_scale (bool): Whether to use class scale. Default: False
        with_cp (bool): Use checkpoint or notnb. Using checkpoint will save some
    """

    def __init__(self,
                 core_op='DCNv3',
                 channels=112,
                 depths=[4, 4, 21, 4],
                 groups=[7, 14, 28, 56],
                 mlp_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.4,
                 drop_path_type='linear',
                 act_layer='GELU',
                 norm_layer='LN',
                 layer_scale=1.0,
                 offset_scale=1.0,
                 post_norm=True,
                 with_cp=False):
        super().__init__()
        self.core_op = core_op
        self.num_levels = len(depths)
        self.depths = depths
        self.channels = channels
        self.num_features = int(channels * 2**(self.num_levels - 1))
        self.post_norm = post_norm
        self.mlp_ratio = mlp_ratio
        print(f'using core type: {core_op}')
        print(f'using activation layer: {act_layer}')
        print(f'using main norm layer: {norm_layer}')
        print(f'using dpr: {drop_path_type}, {drop_path_rate}')
        print(f'using network structure: {cfg.name}')

        in_chans = 3
        
        if cfg.STEMPRO:
            self.patch_embed = Stempro(
                in_chans=in_chans,
                out_chans=channels,
                act_layer='GELU',
                norm_layer='BN'
            )
        else:
            self.patch_embed = StemLayer(in_chans=in_chans,
                                        out_chans=channels,
                                        act_layer=act_layer,
                                        norm_layer=norm_layer)
            
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]
        if drop_path_type == 'uniform':
            for i in range(len(dpr)):
                dpr[i] = drop_path_rate

        self.levels_down = nn.ModuleList()
        for i in range(self.num_levels):
            level_down = InternImageBlock(
                core_op=getattr(opsm, self.core_op),
                channels=int(channels * 2**i),
                depth=depths[i],
                groups=groups[i],
                mlp_ratio=self.mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                act_layer=act_layer,
                norm_layer=norm_layer,
                post_norm=post_norm,
                downsample=(i < self.num_levels - 1),
                layer_scale=layer_scale,
                offset_scale=offset_scale,
                with_cp=with_cp)
            self.levels_down.append(level_down)

        self.levels_up = nn.ModuleList()
        de_depths = depths[::-1]
        de_groups = groups[::-1]
        de_dpr = dpr[::-1]
        for i in range(self.num_levels):
            if i == 0:
                if cfg.PIXEL_SHUFFLE_UP:
                    level_up = PatchExpand(self.channels*2**(self.num_levels-1))
                    
                elif cfg.CBAMDECODER:
                    level_up = CBAMdecoder(
                        self.channels*2**(self.num_levels-1),
                        upsample=True,
                        start=True
                    )
                else:
                    level_up = UpsampleLayer(
                        channels=self.channels*2**(self.num_levels-1),
                        norm_layer=norm_layer
                        )
            else:
                if cfg.CBAMDECODER:
                # ---------for CBAMdecoder--------
                    level_up = CBAMdecoder(int(self.channels * 2 ** (self.num_levels-1-i)), upsample=(i < self.num_levels - 1))
                    
                elif cfg.LAMBDA_UP:
                    
                    level_up = LambdUp(channels = int(self.channels * 2 ** (self.num_levels-1-i)),
                                       num_layer = 3 if i < 2 else 2,
                                       upsample= True if i < 3 else False)
                
                else:
                    level_up = InternImageBlock_up(
                        core_op=getattr(opsm, core_op),
                        channels=int(self.channels * 2 ** (self.num_levels-1-i)),
                        depth=de_depths[i],
                        groups=de_groups[i],
                        mlp_ratio=self.mlp_ratio,
                        drop=drop_rate, 
                        drop_path = de_dpr[sum(de_depths[:i]):sum(de_depths[:i+1])],
                        act_layer=act_layer,
                        post_norm=post_norm,
                        upsample=(i < self.num_levels - 1),
                        layer_scale=layer_scale,
                        offset_scale=offset_scale,
                        with_cp=with_cp)
            self.levels_up.append(level_up)

        if not cfg.AG:
            self.concat_backs = nn.ModuleList()
            for i in range(self.num_levels - 1):
                in_channels = self.channels * 2 ** (self.num_levels - 1 - i)
                out_channels = in_channels//2
                concat_back = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                # build_act_layer('GELU'),
                # build_norm_layer(out_channels, 'LN', 'channels_first', 'channels_first')
                )
                self.concat_backs.append(concat_back)

        # # --------for Attention Gates or Attention Gates Double-------
        if cfg.AG:
            self.AGs = nn.ModuleList()
            for i in range(self.num_levels - 1):
                ag = AttentionGate(self.channels * 2 ** (self.num_levels-2-i), self.channels * 2 ** (self.num_levels-2-i))  # for AG
                self.AGs.append(ag)
                
        # for atttention gate module
        if cfg.AGD:
            self.AGs = nn.ModuleList()
            for i in range(self.num_levels-1):
                ag = AttentionGateD(self.channels * 2 ** (self.num_levels-2-i))   # for AGD
                self.AGs.append(ag)    
                
        # ---------for SideOut -------------
        if cfg.SIDEOUT:
            self.sideouts = nn.ModuleList()
            for i in range(self.num_levels):
                sideout = SideOut(self.channels * 2 ** max(self.num_levels-1-i, 0),
                                2**max(self.num_levels-1-i, 0),
                                self.channels,
                                norm_layer)
                self.sideouts.append(sideout)
        
        # ------for upCBAM   
        if cfg.UPCBAM:
            self.upcbam = UpCBAM(int(((2**self.num_levels-1)/2)*self.channels), 
                                 self.channels, 
                                 norm_layer='BN')
            if cfg.RECTIFICATOIN:
                self.rectification = Rectification(self.channels//2)
            else:
                self.final_output = nn.Conv2d(self.channels//2, 1, 1)
        
        # ------for CBAMup-----------
        if cfg.CBAMUP:
            self.cbamups = nn.ModuleList()
            # self.final_outputs = nn.ModuleList()
            for i in range(self.num_levels):
                cbamup = CBAMup(
                    self.channels * 2 ** max(self.num_levels-1-i, 0),
                    scale=2**max(self.num_levels+1-i, 0),
                    embdchanls=self.channels,
                    num_class=1
                )
                # finaloutput = nn.Conv2d(self.channels//2, 1, 1, 1)
                self.cbamups.append(cbamup)
                # self.final_outputs.append(finaloutput)
            if cfg.RECTIFICATOIN:
                self.rectification = Rectification(self.channels*self.num_levels//2)
            else:
                self.final_output = nn.Conv2d(self.channels*self.num_levels//2, 1, 1, 1)
            # self.final_output = nn.Conv2d(4, 1, 1, 1) 
        #----------for pure DeformConvCrack---------
        self.final_upsample = FinalUpsample(in_channels=self.channels, num_class=1)
            
        self.apply(self._init_weights)
        self.apply(self._init_deform_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)

    def _init_deform_weights(self, m):
        if isinstance(m, getattr(opsm, self.core_op)):
            m._reset_parameters()

    def forward_down_features(self, x):
        x = self.patch_embed(x)
        emd_x = x
        x = self.pos_drop(x)

        down_features = []
        for level_idx, level_down in enumerate(self.levels_down):
            x, x_ = level_down(x, return_wo_downsample=True)
            down_features.append(x_)  
        return down_features, emd_x   

    def forward_up_features(self, downs):
        downs = downs[::-1]
        sideoutputs = []
        x = downs[0]
        if not cfg.CBAMDECODER:
            sideoutputs.append(x)   
        for inx, level_up in enumerate(self.levels_up):
            if inx == 0:
                if cfg.CBAMDECODER:
                    x, _x = level_up(x, sideout=True)
                    sideoutputs.append(_x)
                else:
                    x = level_up(x)
            else:
                #-----for attention gates------
                if cfg.AGD: # for AGD
                    skip_connetion_attention = self.AGs[inx-1](x, downs[inx]) 
                    skip_connection = skip_connetion_attention * downs[inx]
                    x = torch.concat([x, skip_connection], dim=-1)
                    x = self.concat_backs[inx-1](x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
                elif cfg.AG: # for AG
                    x = self.AGs[inx-1](downs[inx], x)
                else:
                    x = torch.concat([x, downs[inx]], dim=-1) 
                    x = self.concat_backs[inx-1](x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
                x, _x = level_up(x, sideout=True)
                sideoutputs.append(_x)
        return x, sideoutputs

    def forward(self, x):
        downs, emd_x = self.forward_down_features(x)
        up, upsamples = self.forward_up_features(downs)
     
        # ---------for SideOut + CBAMup-----
        if cfg.CBAMUP:
            stageoutputs = []
            for inx, cbamup in enumerate(self.cbamups):
                output = cbamup(upsamples[inx])
                # output = self.final_outputs[inx](output)
                stageoutputs.append(output)
                
            if cfg.RECTIFICATOIN:
                out1, attn1 = self.rectification(torch.concat(stageoutputs, 1))
                out2, attn2 = self.rectification(attn1*torch.concat(stageoutputs, 1))
                return out2, out1, torch.ones_like(out1, requires_grad=False), torch.ones_like(out1, requires_grad=False), torch.ones_like(out1, requires_grad=False), torch.ones_like(out1, requires_grad=False)
            else:       
                finaloutput = self.final_output(torch.concat(stageoutputs, 1))
                return finaloutput, torch.ones_like(finaloutput, requires_grad=False), torch.ones_like(finaloutput, requires_grad=False), torch.ones_like(finaloutput, requires_grad=False), torch.ones_like(finaloutput, requires_grad=False), torch.ones_like(finaloutput, requires_grad=False)
                # return finaloutput, torch.ones_like(finaloutput, requires_grad=False), stageoutputs[0], stageoutputs[1], stageoutputs[2], stageoutputs[3]
        
        # ----------for SideOut + UpCBAM-----------
        
        if cfg.SIDEOUT:
            sideoutputs = []
            for inx, so in enumerate(self.sideouts):
                sideoutput = so(upsamples[inx])
                sideoutputs.append(sideoutput)
                
        if cfg.UPCBAM:
            out = torch.concat(sideoutputs, 1)
            out = self.upcbam(out)
            
            if cfg.RECTIFICATOIN:
                out1, attn1 = self.rectification(out)
                out2, attn2 = self.rectification(attn1*out)
                return out2, out1, torch.ones_like(out1, requires_grad=False), torch.ones_like(out1, requires_grad=False), torch.ones_like(out1, requires_grad=False), torch.ones_like(out1, requires_grad=False)
            else:
                finaloutput = self.final_output(out)
                return finaloutput, torch.ones_like(finaloutput, requires_grad=False), torch.ones_like(finaloutput, requires_grad=False), torch.ones_like(finaloutput, requires_grad=False), torch.ones_like(finaloutput, requires_grad=False), torch.ones_like(finaloutput, requires_grad=False)
        

        # ----------pure DeformConvCrack-------
        finaloutput = self.final_upsample(up.permute(0, 3, 1, 2)).permute(0, 3, 1, 2)
        return finaloutput, torch.ones_like(finaloutput, requires_grad=False), torch.ones_like(finaloutput, requires_grad=False), torch.ones_like(finaloutput, requires_grad=False), torch.ones_like(finaloutput, requires_grad=False), torch.ones_like(finaloutput, requires_grad=False)

        
class DeformConvCrack(nn.Module):
    
    def __init__(self, config=cfg):
        super().__init__()
        self.config = config
        self.deformconvcrack = InternImage(
            core_op=config.CORE_OP,
            channels=config.CHANNELS,
            depths=config.DEPTHS,
            groups=config.GROUPS,
            mlp_ratio=config.MLP_RATIO,
            drop_rate=config.DROP_RATE,
            drop_path_rate=config.DROP_PATH_RATE,
            drop_path_type=config.DROP_PATH_TYPE,
            act_layer=config.ACT_LAYER,
            norm_layer=config.NORM_LAYER,
            layer_scale=config.LAYER_SCALE,
            offset_scale=config.OFFSET_SCALE,
            post_norm=config.POST_NORM,
            with_cp=config.WITH_CP)
    def forward(self, x):
        logits, fuse5, fuse4, fuse3, fuse2, fuse1 = self.deformconvcrack(x)
        return logits, fuse5, fuse4, fuse3, fuse2, fuse1 

    def load_from(self, config=cfg):
        pretrain_path = config.PRETRAIN_PATH
        if pretrain_path is not None:
            print("pretrained model path: {}".format(pretrain_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_pth = torch.load(pretrain_path, map_location=device)
            pretrained_dict = pretrained_pth['model']
            print("--start load pretrained model of InterImage encoder---")
            model_dict = self.deformconvcrack.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "levels." in k:
                    down_k = "levels_down." + k[7:]
                    full_dict.update({down_k:v})
                    up_num = len(config.DEPTHS) - int(k[7:8]) - 1
                    up_k = "levels_up." + str(up_num) + k[8:]
                    full_dict.update({up_k:v})
                    del full_dict[k]
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete: {}; pretrained shape:{}; model shape: {}".format(k, v.shape, model_dict[k].shape))
                        del full_dict[k]
                else:
                    pass
                    # print("emitting {}".format(k))
            msg = self.deformconvcrack.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("no pretrain")
            

if __name__ == "__main__":
    
    model = InternImage()
    test = torch.rand(1, 3, 512, 512)
    device = torch.device("cuda")
    model.to(device)
    test = test.cuda()
    out = model(test)
    out = out
    for inx, i in enumerate(out):
        print(str(inx)+"th's shape is: ", i.shape)
    comp, param = get_model_complexity_info(model, (3, 512, 512), as_strings=True, print_per_layer_stat=False, verbose=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', comp))  # GFLOPs = GMac*2
    print('{:<30}  {:<8}'.format('Number of parameters: ', param))
    print("transformer have {}M paramerters in total".format(sum(x.numel() for x in model.parameters())/1000000))
    
    # os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id
    # model = DeformConvCrack(cfg)
    # model.load_from(cfg)
    # device = torch.device("cuda")
    # model.to(device)
    # input = torch.ones([1,3,512,512])
    # input = input.cuda()
    # output = model(input)
    # print(output[0].shape)
    # print(output[1].shape)
    # print(output[2].shape)
    # print(output[3].shape)
    # print(output[4].shape)
    # print(output[5].shape)
    
    # input = torch.rand([1, 128, 128, 128])
    # model = PatchExpand(128, )
    # output = model(input)
    # print(output.shape)
    
    # print(model)
    
    