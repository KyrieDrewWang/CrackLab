import torch
 
import matplotlib.pyplot as plt
 
 
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows
 
 
window_size = 7
shift_size = 3
H, W = 56, 56
img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
h_slices = (slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None))
w_slices = (slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None))
cnt = 0
for h in h_slices:
    for w in w_slices:
        img_mask[:, h, w, :] = cnt
        cnt += 1
 
mask_windows = window_partition(img_mask, window_size)  # nW, window_size, window_size, 1
mask_windows = mask_windows.view(-1, window_size * window_size)
 
attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
 
plt.matshow(img_mask[0, :, :, 0].numpy())
plt.savefig('1.jpg',dpi=600,format='jpg')
plt.matshow(attn_mask[0].numpy())
plt.savefig('2.jpg',dpi=600,format='jpg')
plt.matshow(attn_mask[1].numpy())
plt.savefig('3.jpg',dpi=600,format='jpg')
plt.matshow(attn_mask[2].numpy())
plt.savefig('4.jpg',dpi=600,format='jpg')
plt.matshow(attn_mask[3].numpy())
plt.savefig('5.jpg',dpi=600,format='jpg')
plt.show()