import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
       
class FFN(nn.Module):
    def __init__(self, 
                dims, 
                mlp_ratio,
                act_layer, 
                drop_rate,
                drop_path_rate,
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
        if identity == None:
            identity = x
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.drop(out)
        out = self.droppath(out) + identity
        return out


class Bridege(nn.Module):
    def __init__(self, 
                dim,
                mlp_ratio,
                act_layer, 
                drop_rate,
                drop_path_rate, ) -> None:
        super().__init__()
        self.dim = dim
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        # self.depthwise = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        # self.pointwise = nn.Conv2d(dim, dim, kernel_size=1)
        self.bnorm = nn.BatchNorm2d(dim)
        self.act_layer = nn.GELU()
        self.layer_norm = nn.LayerNorm(dim)
        self.ffn = FFN(dim, mlp_ratio, act_layer, drop_rate, drop_path_rate)
    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H*W, "invalid input size"
        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        skip_x = x
        # x = self.depthwise(x)
        # x = self.pointwise(x)
        x = self.conv(x)
        x = self.bnorm(x)
        x = self.act_layer(x)
        x = skip_x + x
        x = x.permute(0,2,3,1)
        x = x.view(B, H*W, C)
        x = self.layer_norm(x)
        out = self.ffn(x)
        return out


class Fuse(nn.Module):  
    def __init__(self, channels, embed_dims, scale) -> None:
        super().__init__()
        self.channels = channels
        self.embed_dims = embed_dims
        self.scale = scale
        self.conv1 = nn.Conv2d(channels, self.embed_dims, kernel_size=3, padding=1, stride=1)
        self.norm = nn.LayerNorm(self.embed_dims)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(self.embed_dims, 1, kernel_size=3, padding=1, stride=1)
    def forward(self, x):
        """
        input:  [B, H, W, C]
        output: [B, C, H, W]
        """
        x = x.permute(0,3,1,2)
        out = F.interpolate(out, scale_factor=self.scale, mode='bilinear')
        out = self.conv1(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.conv2(out)
        return out



class CALayer(nn.Module):
    '''
    channel fusion module
    '''
    def __init__(self, num_channels, reduction_ratio):
        super(CALayer, self).__init__()
        '''
        :param num_channels: number of input channels
        :param reduction_ratio: ratio applied to reduce the num_channels(reduce model complexity)
        '''
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        '''

        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor(fusion of all the channels)
        '''
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)
        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor

class SALayer(nn.Module):  # Spatial Attention Layer
    def __init__(self):
        super().__init__()
        self.conv_sa = nn.Sequential(
            nn.Conv2d(2, 1, 3, 1, 1, bias=False),
            nn.BatchNorm2d(1, momentum=0.01),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_compress = torch.cat(
            (torch.max(x, 1, keepdim=True)[0], torch.mean(x, dim=1, keepdim=True)), dim=1)
        scale = self.conv_sa(x_compress)
        return x * scale


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=2, pool_types=['avg', 'max']):
        super().__init__()
        self.CALayer = CALayer(
            in_channels, reduction, pool_types)
        self.SALayer = SALayer()

    def forward(self, x):
        x_out = self.CALayer(x)
        x_out = self.SALayer(x_out)
        return x_out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
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
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)



class Gatattention(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Gatattention, self).__init__()
        self.sigma = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(output_channels//4,output_channels)
        )
        self.fi = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(output_channels//4,output_channels),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        self.gelu=nn.GELU()
    def forward(self, inputs):
        sum = 0
        for input in inputs:
            sum += input
        sum=self.gelu(sum)
        out = self.sigma(sum)
        att = self.fi(out)  # Mask
        return att
    

# input: (B, L=HxW, C)
class nolocalExpand(nn.Module):
    def __init__(self, 
                 num_heads,
                 emd_dims,  # input dims
                 qkv_bias=True, 
                 qk_scale=None, 
                 attn_drop=0., 
                 proj_drop=0.,
                 drop_path_rate=0.,
                 act_layer=nn.GELU,
                 ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.emd_dims = emd_dims // 2
        self.kv_reduction=nn.Linear(emd_dims, self.emd_dims)
        self.proj_q = nn.Linear(self.emd_dims, self.emd_dims, bias=qkv_bias)
        self.proj_kv =nn.Linear(self.emd_dims, emd_dims, bias=qkv_bias)
        emd_dims_per_head = self.emd_dims // self.num_heads
        self.scale = qk_scale or emd_dims_per_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_proj = nn.Linear(self.emd_dims, self.emd_dims)
        self.proj_drop = nn.Dropout(proj_drop)
        self.drop_path = DropPath(drop_path_rate)
        self.skip_atten_linear = nn.Linear(emd_dims, self.emd_dims)
        self.attn_norm = nn.LayerNorm(self.emd_dims)
        
        self.ffn = Mlp(self.emd_dims, 4*self.emd_dims, act_layer=act_layer, drop=proj_drop)
        self.ffn_norm = nn.LayerNorm(self.emd_dims)
        self.outffn = nn.Linear(2*self.emd_dims, self.emd_dims)
    def forward(self, x, skip_x):  # H, W --> input size        
        _, H, W, C = x.shape
        x = x.view(-1, H*W, C)
        _, h, w, c = skip_x.shape
        skip_x = skip_x.view(-1, h*w, c)
        
        skip_attn = x
        #Nolocal upsampling
        x = self.kv_reduction(x)
        B, L, C = x.shape
        b, l, c = skip_x.shape
        assert C == c, "input and skip have unequal dimensions"
        q = self.proj_q(skip_x).reshape(b, l, 1, self.num_heads, c // self.num_heads).permute(2,0,3,1,4)
        kv = self.proj_kv(x).reshape(B, L, 2, self.num_heads, c // self.num_heads).permute(2,0,3,1,4)
        q, k, v = q[0], kv[0], kv[1]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, l, C)
        x = self.attn_proj(x)
        x = self.proj_drop(x)

        #skip attn connection
        bs, ls, cs = skip_attn.shape
        assert ls == H*W, "input size wrong"
        skip_attn = skip_attn.view(bs, H, W, cs).permute(0, 3, 1, 2)
        skip_attn = F.interpolate(skip_attn, scale_factor=2, mode='bilinear')
        bs, cs, H, W  = skip_attn.shape
        skip_attn = skip_attn.permute(0, 2, 3, 1).view(bs, H*W, cs)
        skip_attn = self.skip_atten_linear(skip_attn)
        x = self.drop_path(x) + skip_attn
        
        #FFN
        skip_ffn = x
        x = self.attn_norm(x)
        x = skip_ffn + self.drop_path(self.ffn(x))
        x = self.ffn_norm(x)
        out = torch.concat([x, skip_x], -1)
        out = self.outffn(out)
        return x.view(B, H, W, c)



if __name__ == "__main__":
    skip_x = torch.rand([1, 32*32, 4])
    x = torch.rand([1, 16*16, 8])
    model = nolocalExpand(2,
                          8,
                          qkv_bias=True,
                          qk_scale=None,
                          )
    out = model(x, skip_x, 16, 16)
    print(out.shape)
        
        