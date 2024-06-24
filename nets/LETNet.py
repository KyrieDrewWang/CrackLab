import torch
import math
# from kornia import canny
import torch.nn as nn
from ptflops import get_model_complexity_info
import sys
sys.path.append('.')
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from thop import profile
from config.config_LETNet import Config as cfg

class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 sr_ratio=1,
                 apply_transform=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio+1, stride=sr_ratio, padding=sr_ratio // 2, groups=dim)
            self.sr_norm = nn.LayerNorm(dim)

        self.apply_transform = apply_transform and num_heads > 1
        if self.apply_transform:
            self.transform_conv = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1, stride=1)
            self.transform_norm = nn.InstanceNorm2d(self.num_heads)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.sr_norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.apply_transform:
            attn = self.transform_conv(attn)
            attn = attn.softmax(dim=-1)
            attn = self.transform_norm(attn)
        else:
            attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)  # ########
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)  # ########
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, apply_transform=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, apply_transform=apply_transform)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pa_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.pa_conv(x))


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, in_ch=3, out_ch=768, with_pos=False):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=patch_size+1, stride=patch_size, padding=patch_size // 2)
        self.norm = nn.BatchNorm2d(out_ch)

        self.with_pos = with_pos
        if self.with_pos:
            self.pos = PA(out_ch)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv(x)
        x = self.norm(x)

        if self.with_pos:
            x = self.pos(x)
        x_flatten = x.flatten(2).transpose(1, 2)
        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, x_flatten, (H, W)


class PatchEmbed_decoder(nn.Module):
    def __init__(self, patch_size=16, in_ch=3, out_ch=768, with_pos=False):
        super().__init__()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x, (H, W)


class BasicStem(nn.Module):
    def __init__(self, in_ch=3, out_ch=64, with_pos=False) -> object:
        super(BasicStem, self).__init__()
        hidden_ch = out_ch // 2
        self.conv1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(hidden_ch)
        self.conv2 = nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(hidden_ch)
        self.conv3 = nn.Conv2d(hidden_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)

        self.act = nn.ReLU(inplace=True)
        self.with_pos = with_pos
        if self.with_pos:
            self.pos = PA(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)

        x = self.conv3(x)
        if self.with_pos:
            x = self.pos(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels=768, out_channels=768, factor=2):
        super().__init__()
        self.up = nn.Sequential(
            nn.PixelShuffle(upscale_factor=factor),  # channel out = channel in / upscale_factor**2
            nn.Conv2d(in_channels//factor**2, out_channels, kernel_size=1),  # 换成3*3卷积呢
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


class LastConv(nn.Module):
    def __init__(self, in_ch):
        super(LastConv, self).__init__()
        self.conv = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(in_ch//4, out_channels=in_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2),
            nn.Conv2d(in_ch//4, out_channels=in_ch//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_ch//2),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Predict(nn.Module):
    def __init__(self, in_ch=32):
        super(Predict, self).__init__()
        self.predict = nn.Conv2d(in_ch, 1, kernel_size=1)

    def forward(self, x):
        x = self.predict(x)
        attn = torch.sigmoid(x)
        return x, attn


class Residual(nn.Module):

    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=False),  # in_ch, int(in_ch*2)
            nn.BatchNorm2d(in_ch),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class LETNet(nn.Module):
    def __init__(self, 
                 in_chans=cfg.IN_CHANS, 
                 embed_dims=cfg.EMBED_DIMS,
                 num_heads=cfg.NUM_HEADS, 
                 mlp_ratios=cfg.MLP_RATIOS, 
                 qkv_bias=cfg.QKV_BIAS,
                 qk_scale=cfg.QK_SCALE, 
                 drop_rate=cfg.DROP_RATE, 
                 attn_drop_rate=cfg.ATTN_DROP_RATE, 
                 drop_path_rate=cfg.DROP_PATH_RATE,
                 depths=cfg.DEPTHS, 
                 sr_ratios=cfg.SR_RATIOS,
                 norm_layer=nn.LayerNorm, 
                 apply_transform=True):
        super().__init__()
        self.depths = depths
        self.apply_transform = apply_transform

        self.stem = BasicStem(in_ch=in_chans, out_ch=embed_dims[0], with_pos=True)

        self.patch_embed_2 = PatchEmbed(patch_size=2, in_ch=embed_dims[0], out_ch=embed_dims[1], with_pos=False)  # 下采样倍数
        self.patch_embed_3 = PatchEmbed(patch_size=2, in_ch=embed_dims[1], out_ch=embed_dims[2], with_pos=False)
        self.patch_embed_4 = PatchEmbed(patch_size=2, in_ch=embed_dims[2], out_ch=embed_dims[3], with_pos=False)
        self.patch_embed_decoder_2 = PatchEmbed_decoder(patch_size=1, in_ch=int(embed_dims[1]), out_ch=embed_dims[1], with_pos=False)
        self.patch_embed_decoder_3 = PatchEmbed_decoder(patch_size=1, in_ch=int(embed_dims[2]), out_ch=embed_dims[2], with_pos=False)
        self.patch_embed_decoder_4 = PatchEmbed_decoder(patch_size=1, in_ch=int(embed_dims[0]), out_ch=embed_dims[0], with_pos=False)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        self.stage1 = nn.ModuleList([
            Block(embed_dims[0], num_heads[0], mlp_ratios[0], qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                  drop_path=dpr[cur+i], norm_layer=norm_layer, sr_ratio=sr_ratios[0], apply_transform=apply_transform)
            for i in range(self.depths[0])])

        cur += depths[0]
        self.stage2 = nn.ModuleList([
            Block(embed_dims[1], num_heads[1], mlp_ratios[1], qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                  drop_path=dpr[cur+i], norm_layer=norm_layer, sr_ratio=sr_ratios[1], apply_transform=apply_transform)
            for i in range(self.depths[1])])

        cur += depths[1]
        self.stage3 = nn.ModuleList([
            Block(embed_dims[2], num_heads[2], mlp_ratios[2], qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                  drop_path=dpr[cur+i], norm_layer=norm_layer, sr_ratio=sr_ratios[2], apply_transform=apply_transform)
            for i in range(self.depths[2])])

        cur += depths[2]
        self.stage4 = nn.ModuleList([
            Block(embed_dims[3], num_heads[3], mlp_ratios[3], qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                  drop_path=dpr[cur+i], norm_layer=norm_layer, sr_ratio=sr_ratios[3], apply_transform=apply_transform)
            for i in range(self.depths[3])])

        self.norm = norm_layer(embed_dims[3])

        self.up1 = Up(in_channels=embed_dims[3], out_channels=embed_dims[3]//2)
        self.up2 = Up(in_channels=embed_dims[3]//2, out_channels=embed_dims[3]//4)
        self.up3 = Up(in_channels=embed_dims[3]//4, out_channels=embed_dims[3]//8)
        self.residual1 = Residual(embed_dims[0])
        self.residual2 = Residual(embed_dims[1])
        self.residual3 = Residual(embed_dims[2])
        self.residual4 = Residual(embed_dims[3])
        self.lastconv = LastConv(in_ch=embed_dims[3]//8)
        self.predict = Predict(48)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02).cuda()
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02).cuda()
            if m.bias is not None:
                nn.init.constant_(m.bias, 0).cuda()
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1.0).cuda()
            nn.init.constant_(m.bias, 0).cuda()

    def forward(self, x):
        x0 = self.stem(x)
        B, _, H, W = x0.shape
        x = x0.flatten(2).permute(0, 2, 1)

        # stage 1
        for blk in self.stage1:
            x = blk(x, H, W)
        x1_t = x.permute(0, 2, 1).reshape(B, -1, H, W)  # [b, c*64, h/4, w/4] [4 64 56 56]
        x1 = self.residual1(x1_t)

        # stage 2
        x_2, x, (H, W) = self.patch_embed_2(x1)
        for blk in self.stage2:
            x = blk(x, H, W)
        x2_t = x.permute(0, 2, 1).reshape(B, -1, H, W)  # [b, c*128, h/8, w/8] [4 128 28 28]
        x2 = self.residual2(x2_t)

        # stage 3
        x_3, x, (H, W) = self.patch_embed_3(x2)
        for blk in self.stage3:
            x = blk(x, H, W)
        x3_t = x.permute(0, 2, 1).reshape(B, -1, H, W)  # [b, c*256, h/16, w/16]
        x3 = self.residual3(x3_t)

        # stage 4
        x_4, x, (H, W) = self.patch_embed_4(x3)
        for blk in self.stage4:
            x = blk(x, H, W)
        x = self.norm(x)
        x4_t = x.permute(0, 2, 1).reshape(B, -1, H, W)  # [b, c*512, h/32, w/32]
        x4 = self.residual4(x4_t)

        # decoder 1
        x = self.up1(x4)
        x = x+x3+x3_t

        x, (W, H) = self.patch_embed_decoder_3(x)
        # H, W = 14, 14
        for blk in self.stage3:
            x = blk(x, H, W)
        decoder1 = x.permute(0, 2, 1).reshape(B, -1, H, W)
        decoder1 = self.residual3(decoder1)

        # decoder 2
        x = self.up2(decoder1)
        x = x+x2+x2_t
        # H, W = 28, 28
        x, (W, H) = self.patch_embed_decoder_2(x)
        for blk in self.stage2:
            x = blk(x, H, W)
        decoder2 = x.permute(0, 2, 1).reshape(B, -1, H, W)
        decoder2 = self.residual2(decoder2)

        # decoder 3
        x = self.up3(decoder2)
        x = x+x1+x1_t
        x, (W, H) = self.patch_embed_decoder_4(x)
        for blk in self.stage1:
            x = blk(x, H, W)
        decoder3 = x.permute(0, 2, 1).reshape(B, -1, H, W)
        decoder3 = self.residual1(decoder3)

        # output
        pred_feature = self.lastconv(decoder3)
        pred_1, attn1 = self.predict(pred_feature)
        pred_2, attn2 = self.predict(attn1 * pred_feature)

        return pred_2, pred_1, torch.ones_like(pred_1, requires_grad=False), torch.ones_like(pred_1, requires_grad=False), torch.ones_like(pred_1, requires_grad=False), torch.ones_like(pred_1, requires_grad=False)

if __name__ == '__main__':
    model = LETNet()#.cuda()
    # inputs = torch.rand(1, 1, 224, 224).cuda()
    # macs, params = get_model_complexity_info(model, (1, 224, 224), as_strings=True,
    #                                          print_per_layer_stat=False, verbose=False)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))  # GFLOPs = GMac*2
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # print("transformer have {}M paramerters in total".format(sum(x.numel() for x in model.parameters())/1000000))
    inputs = torch.rand(1,3,224,224)
    a, b, b, b, b, b = model(inputs)
    print(a.shape)

