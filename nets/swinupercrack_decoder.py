import os
import sys
sys.path.append('.')
import torch
import torch.nn as nn
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops.layers.torch import Rearrange
from config._config import baseconfig as bcfg
from nets.swincrack_backbone import DSF


class FFN(nn.Module):
    def __init__(self, 
                dims, 
                mlp_ratio,
                act_layer, 
                drop_rate,
                drop_path_rate
                ) -> None:
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dims, int(dims*mlp_ratio)),
            act_layer(),
            nn.Dropout(drop_rate)
        )
        self.fc2 = nn.Linear(int(mlp_ratio*dims), dims)
        self.drop = nn.Dropout(drop_rate) if drop_rate>0 else nn.Identity()
        self.droppath = DropPath(drop_path_rate) if drop_path_rate>0 else nn.Identity()
    def forward(self, x, identity=None):
        if identity is None:
            identity = x
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.drop(out)
        out = self.droppath(out) + identity
        return out


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        # x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class WindowMSA(nn.Module):
    """Window based multi-head self-attention (W-MSA) module with relative
    position bias.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.
        init_cfg (dict | None, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.):
        super().__init__()
        self.embed_dims = embed_dims
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # About 2x faster than original impl
        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(step1=(2 * Ww - 1), len1=Wh, step2=1, len2=Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)

        if bcfg.DE_use_convattn:
            # substitue self.qkv for conv proj in CST
            self.conv_proj_q = self._build_convprojection(embed_dims, kernel_size=3, stride=1, padding=1)
            self.conv_proj_k = self._build_convprojection(embed_dims, kernel_size=3, stride=1, padding=1)
            self.conv_proj_v = self._build_convprojection(embed_dims, kernel_size=3, stride=1, padding=1)      
        else:
            self.qkv = nn.Linear(embed_dims, embed_dims * 2, bias=qkv_bias)
            self.skip_qkv = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)
            
        
        self.attn_drop = nn.Dropout(attn_drop_rate)
        
        if bcfg.DE_use_convproj:
            # substitute self.proj for depth conv in CST
            self.proj = nn.Sequential(
                nn.Conv2d(embed_dims, 
                          embed_dims, 
                          kernel_size=3, 
                          padding=1, 
                          stride=1, 
                          bias=False, 
                          groups=embed_dims),
                nn.GELU()
            )
        else:
            self.proj = nn.Linear(embed_dims, embed_dims)
        
        
        
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)

        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def _build_convprojection(self, dim_in, kernel_size=3, stride=1, padding=1):
        conv_proj = nn.Sequential(
            nn.Conv2d(dim_in, 
                      dim_in, 
                      kernel_size, 
                      padding=padding, 
                      stride=stride, 
                      bias=False, 
                      groups=dim_in),
            Rearrange('b c h w -> b (h w) c'),
            nn.LayerNorm(dim_in)
        )
        return conv_proj

    def forward(self, x, skip_x, mask=None):
        """
        Args:
            x (tensor): input features with shape of (num_windows*B, N, C)
            skip_x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor | None, Optional): mask with shape of (num_windows,
                Wh*Ww, Wh*Ww), value should be between (-inf, 0].
        """
        B, N, C = x.shape
        assert x.shape == skip_x.shape, 'x.shape != skip_x.shape in WindowMSA'
        
        if bcfg.DE_use_convattn:
            # # substitute mlp for conv proj to in CST
            win_size = int(N ** 0.5)
            x = x.view(B, win_size, win_size, C)  # B * reshaped 2D feature map size * reshaped 2D feature map size * C
            x = x.permute(0, 3, 1, 2)  # B * C * reshaped 2D feature map size * reshaped 2D feature map size
            skip_x = skip_x.view(B, win_size, win_size, C).permute(0, 3, 1, 2)
            
            q = self.conv_proj_q(skip_x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  
            k = self.conv_proj_k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = self.conv_proj_v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        
        else:
            qkv = self.qkv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            skip_qkv = self.skip_qkv(skip_x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = skip_qkv[0], qkv[0], qkv[1]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        if bcfg.DE_use_convproj:
            # substitute mlp proj for depth conv in CST
            x = (attn @ v).transpose(2, 3).reshape(B, C, win_size, win_size)
            x = self.proj(x)
            x = x.reshape(B, C, N).transpose(1, 2)
            x = self.proj_drop(x)  
        else:
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
        return x
        
    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


class ShiftWindowMSA(nn.Module):
    """Shifted Window Multihead Self-Attention Module.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window.
        shift_size (int, optional): The shift step of each window towards
            right-bottom. If zero, act as regular window-msa. Defaults to 0.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Defaults: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Defaults: 0.
        proj_drop_rate (float, optional): Dropout ratio of output.
            Defaults: 0.
        dropout_layer (dict, optional): The dropout_layer used before output.
            Defaults: dict(type='DropPath', drop_prob=0.).
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0,
                 proj_drop_rate=0,
                 drop_path=0.):
        super().__init__()

        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.w_msa = WindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, query, skip_query, hw_shape):
        B, L, C = query.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'
        assert query.shape == skip_query.shape, 'skip query should has the same shape with query'
        query = query.view(B, H, W, C)
        skip_query = skip_query.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        skip_query = F.pad(skip_query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]

        # cyclic shift
        if self.shift_size > 0:
            shifted_query = torch.roll(
                query,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2))
            shifted_skip_query = torch.roll(
                skip_query,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2))

            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.view(
                -1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            shifted_query = query
            shifted_skip_query = skip_query
            attn_mask = None

        # nW*B, window_size, window_size, C
        query_windows = self.window_partition(shifted_query)
        skip_query_windows = self.window_partition(shifted_skip_query)
        # nW*B, window_size*window_size, C
        query_windows = query_windows.view(-1, self.window_size ** 2, C)
        skip_query_windows = skip_query_windows.view(-1, self.window_size ** 2, C)

        # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
        attn_windows = self.w_msa(query_windows, skip_query_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)

        # B H' W' C
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        x = self.drop_path(x)
        return x

    def window_reverse(self, windows, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, H, W, C)
        """
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size,
                         window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def window_partition(self, x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows


class SwinBlock(nn.Module):
    """"
    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        window_size (int, optional): The local window scale. Default: 7.
        shift (bool, optional): whether to shift window or not. Default False.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float, optional): Stochastic depth rate. Default: 0.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size=7,
                 shift=False,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU):

        super(SwinBlock, self).__init__()
        self.embed_dims = embed_dims

        self.skip_norm = norm_layer(embed_dims)
        self.norm1 = norm_layer(embed_dims)
        self.attn = ShiftWindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2 if shift else 0,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            drop_path=drop_path_rate)

        self.norm2 = norm_layer(embed_dims)

        if bcfg.DE_use_DSF:
            # substitute self.mlp for DSF
            self.dsf = DSF(dim=embed_dims, drop_path=drop_path_rate)
        else:
            self.ffn = FFN(
                dims=embed_dims,
                mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate
            )

        self.norm3 = norm_layer(embed_dims)

    def forward(self, x, skip_x, hw_shape):
        identity = x
        skip_x = self.skip_norm(skip_x)
        x = self.norm1(x)
        x = self.attn(x, skip_x, hw_shape)
        x = x + identity
        
        identity = x
        x = self.norm2(x)
        
        if bcfg.DE_use_DSF:
            # substitute FFN for DSF
            B, L, C = x.shape
            assert L == hw_shape[0]*hw_shape[1], "input feature has wrong size"
            H, W = hw_shape[0], hw_shape[1]
            x = x.view(B, H, W, C)
            x = self.dsf(x)
            x = x.view(B, H*W, C)
        else:
            x = self.ffn(x, identity=identity)
        x = self.norm3(x)
        return x


class SwinUperBlock(nn.Module):
    """Implements one stage in Swin Transformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        depth (int): The number of blocks in this stage.
        window_size (int, optional): The local window scale. Default: 7.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float | list[float], optional): Stochastic depth
            rate. Default: 0.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
        is_upsample (bool): Whether to apply Fuse&Upsample block.
    """

    def __init__(self,
                embed_dims,
                num_heads,
                depth,
                window_size=7,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.,
                mlp_ratio=4., 
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU,
                is_upsample=True):
        super().__init__()

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = SwinBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                window_size=window_size,
                shift=False if i % 2 == 0 else True,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
                act_layer=act_layer)
            self.blocks.append(block)
        self.is_upsample = is_upsample
        self.conv = nn.Sequential(
                            self._Conv1(in_=embed_dims * 2, out=embed_dims * 2),
                            self._Syncnorm(embed_dims * 2),
                            nn.GELU())
        self.ps = nn.PixelShuffle(2)
        self.convout = nn.Sequential(
                            self._Conv1(in_=embed_dims * 2, out=embed_dims * 2),
                            self._Syncnorm(embed_dims * 2),
                            nn.GELU())
    @staticmethod
    def _Conv1(in_, out):
        conv = nn.Conv2d(in_channels=in_, out_channels=out, kernel_size=1, stride=1, bias=False)
        nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
        return conv

    @staticmethod
    def _Syncnorm(c):
        norm = nn.BatchNorm2d(c)
        nn.init.constant_(norm.bias, 0)
        nn.init.constant_(norm.weight, 1)
        return norm

    def forward(self, x, skip_x, hw_shape):
        for block in self.blocks:
            # if skip_x==None:
            #     skip_x = x
            x = block(x, x, hw_shape)

        if self.is_upsample:
            out = torch.cat([x, skip_x], dim=2)
            up_hw_shape = [hw_shape[0] * 2, hw_shape[1] * 2]
            B, HW, C = out.shape
            out = out.view(B, hw_shape[0], hw_shape[1], C)
            out = out.permute(0, 3, 1, 2)
            out = self.conv(out)
            out = self.ps(out)
            out = out.permute(0, 2, 3, 1).view(B, up_hw_shape[0] * up_hw_shape[1], C // 4)
            return out
        else:
            x = torch.cat([x, skip_x], dim=2)
            B, HW, C = x.shape
            x = x.view(B, hw_shape[0], hw_shape[1], C).permute(0, 3, 1, 2)
            x = self.convout(x)
            # x = self.ps(x)
            x = x.permute(0, 2, 3, 1).view(B, hw_shape[0] * hw_shape[1], C)
            return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
 
    