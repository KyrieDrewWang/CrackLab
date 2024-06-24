import cv2
import numpy as np
import torch
import os
import sys
sys.path.append('.')
sys.path.append('nets')
from nets.swincrack import swincrack
from config.config_swincrack import Config as cfg
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, ToTensor, transforms
import math
from tqdm import tqdm
from data.dataset import readIndex
import torch.nn.functional as F

class SaveHook:
    def __init__(self) -> None:
        super().__init__()
        self.out = []   
             
    def __call__(self, module, module_in, module_out):
        self.out.append(module_out)
                
    def clear(self):
        self.out = []


def preprocess_image(img: np.ndarray, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) -> torch.Tensor:
    preprocessing = Compose([
        ToTensor(),
        # Normalize(mean=mean, std=std)
    ])
    return preprocessing(img.copy()).unsqueeze(0)

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


if __name__ == '__main__':
    

    trained_model = "checkpoints/SwinCrack-abl/CrackLS315-SwinCrack-ConvEmbd-ConvAttn-DFN52-agsc/CrackLS315-SwinCrack-ConvEmbd-ConvAttn-DFN52-agsc_epoch(128)_acc(0.65104_0.65104)_0000040_2023-05-10-21-12-39.pth"
    base_dir = "visual/visualization/swin_ConvEmbd_ConvAttn_DFN52_agsc_mlp"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id
    device = torch.device("cuda")
    img_ann_list = readIndex(cfg.test_data_path)
    model = swincrack()
    model.load_state_dict(torch.load(trained_model, map_location=device), strict=True)
    model = model.to(device=device)
    model.eval()
    bar = tqdm(img_ann_list, desc="visualizing mlp:", ncols=100, unit="fps")
    for img_path, ann_path in bar:
        torch.cuda.empty_cache() 
        img_name = img_path.split('/')[-1]
        img_save_path = os.path.join(base_dir, img_name)
        rgb_img_ = cv2.imread(img_path)
        rgb_img = np.float32(rgb_img_) / 255
        input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        input_tensor = input_tensor.type(torch.cuda.FloatTensor).to(device)
        savehook = SaveHook()
        hk = model.swin_unet.patch_embed.norm.register_forward_hook(savehook)
        output = model(input_tensor)
        hk.remove
        feature_map = savehook.out[0].reshape(-1, 128, 128, 128)
        feature_map = feature_map.permute(0, 3, 1, 2)
        feature_map = F.interpolate(feature_map, scale_factor=4, mode="bilinear")
        feature_map = feature_map.mean(axis=1).squeeze(0)
        feature_map = feature_map.detach().cpu().numpy()
        img = show_mask_on_image(rgb_img_, feature_map)
        cv2.imwrite(img_save_path, img)