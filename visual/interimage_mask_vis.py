import torch
import torch.nn as nn
import sys
sys.path.append('.')
sys.path.append('nets')
from nets.DeformConvCrack import DeformConvCrack
from config.config_DeformConvCrack import Config as cfg
import os
import cv2
import numpy as np
from torchvision.transforms import Compose, Normalize, ToTensor
import copy
from data.dataset import dataReadPip, loadedDataset, readIndex
from tqdm import tqdm

class SaveHook:
    def __init__(self) -> None:
        super().__init__()
        self.out = []   
             
    def __call__(self, module, module_in, module_out):
        self.out.append(module_out)
                
    def clear(self):
        self.out = []
        

def GenerateOffsets(savehook):
    offset_outputs = []
    for inx, depths in enumerate(cfg.DEPTHS):
        temp_output = []
        for depth in range(depths):
            temp_output.append(savehook.out[sum(cfg.DEPTHS[:inx])+depth])
        offset_outputs.append(temp_output)
    return offset_outputs
        
def plot_offsets(img, img_list, save_output, roi_x, roi_y, offset_scale=cfg.OFFSET_SCALE):
    input_img_h, input_img_w = img.shape[:2]
    for inx, save_out in enumerate(save_output):   
        for _inx, offsets in enumerate(save_out):
            b, h, w, c = offsets.shape
            # spatial_norm = torch.tensor([w, h]).reshape(1, 1, 1, 2).repeat(1,1,1,c//2).to(offsets.device)
            # offsets = offsets / spatial_norm
            offsets = offsets.permute(0, 3, 1, 2)
            offsets = offsets.view(b, (c//18), 18, h, w)
            offsets = torch.mean(offsets, dim=1).unsqueeze(1)
            # offsets = torch.max(offsets, dim=1).unsqueeze(1)
            for offset in offsets:
                offsets = offset
                offsets = offsets*offset_scale
                offset_tensor_h, offset_tensor_w = offsets.shape[2:]
                resize_factor_h, resize_factor_w = input_img_h/offset_tensor_h, input_img_w/offset_tensor_w

                offsets_y = offsets[:, ::2]
                offsets_x = offsets[:, 1::2]

                grid_y = np.arange(0, offset_tensor_h)
                grid_x = np.arange(0, offset_tensor_w)

                grid_x, grid_y = np.meshgrid(grid_x, grid_y)

                sampling_y = grid_y + offsets_y.detach().cpu().numpy()
                sampling_x = grid_x + offsets_x.detach().cpu().numpy()

                sampling_y *= resize_factor_h
                sampling_x *= resize_factor_w

                sampling_y = sampling_y[0] # remove batch axis
                sampling_x = sampling_x[0] # remove batch axis

                sampling_y = sampling_y.transpose(1, 2, 0) # c, h, w -> h, w, c
                sampling_x = sampling_x.transpose(1, 2, 0) # c, h, w -> h, w, c

                sampling_y = np.clip(sampling_y, 0, input_img_h)
                sampling_x = np.clip(sampling_x, 0, input_img_w)

                sampling_y = cv2.resize(sampling_y, dsize=None, fx=resize_factor_w, fy=resize_factor_h)
                sampling_x = cv2.resize(sampling_x, dsize=None, fx=resize_factor_w, fy=resize_factor_h)

                sampling_y = sampling_y[roi_y, roi_x]
                sampling_x = sampling_x[roi_y, roi_x]
                for y, x in zip(sampling_y, sampling_x):
                    y = round(y)
                    x = round(x)
                    cv2.circle(img, center=(x, y), color=(0, 0, 255), radius=1, thickness=-1)   
                    cv2.circle(img_list[inx], center=(x, y), color=(0, 0, 255), radius=1, thickness=-1) 
              
    cv2.circle(img, center=(roi_x, roi_y), color=(0, 255, 0), radius=2, thickness=-1)
    for i in range(len(img_list)):  
        cv2.circle(img_list[i], center=(roi_x, roi_y), color=(0, 255, 0), radius=2, thickness=-1)

def preprocess_image(
    img: np.ndarray, mean=[
        0.5, 0.5, 0.5], std=[
            0.5, 0.5, 0.5]) -> torch.Tensor:
    preprocessing = Compose([
        ToTensor(),
        # Normalize(mean=mean, std=std)
    ])
    return preprocessing(img.copy()).unsqueeze(0)


if __name__ == "__main__":
    trained_model = "checkpoints/CrackLS315-DeformConvCrack-offset2-AGD-CBAMup-droprate0.1/CrackLS315-DeformConvCrack-offset2-AGD-CBAMup-droprate0.1_epoch(29)_acc(0.64331_0.64331)_0000022_2023-04-10-22-48-14.pth"
    base_dir = "visual/visualization/offset_vis"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id
    device = torch.device("cuda")

    model = DeformConvCrack(cfg)
    model.load_state_dict(torch.load(trained_model))
    model = model.cuda()
    model.eval()
    # print(model)

    img_ann_path = readIndex(cfg.test_data_path)
    bar = tqdm(img_ann_path, desc="visualizing offset:", ncols=100, unit="fps")
    with torch.no_grad():    
        for img_path, ann_path in bar:
            
            img_name = img_path.split('/')[-1].split('.')[0]
            img_dir = os.path.join(base_dir, img_name)
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)
            
            rgb = cv2.imread(img_path)
            ann = cv2.imread(ann_path, 0)
            
            h, w, _ = rgb.shape 
            input_tensor = np.float32(rgb) / 255
            input_tensor = preprocess_image(input_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            input_tensor = input_tensor.type(torch.cuda.FloatTensor).to(device)

            savehook = SaveHook()
            for inx, depths in enumerate(cfg.DEPTHS):
                for depth in range(depths):
                    hk = model.deformconvcrack.levels_down[inx].blocks[depth].dcn.mask.register_forward_hook(savehook)

            output = model(input_tensor)
            hk.remove()
            
            masks_outputs = GenerateOffsets(savehook)
            
            for inx, save_out in enumerate(masks_outputs): 
                  
                for _inx, masks in enumerate(save_out):
                    b, h, w, c = masks.shape
                    masks = masks.permute(0,3,1,2)
                    masks = masks.view(b, c//9, 9, h, w)
                    masks = torch.mean(masks, dim=1).unsqueeze(1)
                    for mask in masks:
                        mask_h, mask_w = mask.shape[2:]
                        resize_factor_h, resize_factor_w = cfg.IMG_SIZE / mask_h, cfg.IMG_SIZE / mask_w
                        mask = torch.sum(mask, dim=1)
                        mask = mask[0]
                        mask = cv2.resize(mask, dsize=None, fx=resize_factor_w, fy=resize_factor_h)
                        
                        
            

            final_img_path = os.path.join(img_dir, img_name+'_final.jpg')
            cv2.imwrite(final_img_path, rgb)
            
            for inx, img in enumerate(img_list):
                img_path = os.path.join(img_dir, img_name+'_'+str(inx)+'.jpg')
                cv2.imwrite(img_path, img)
                
            savehook.clear()
        bar.close()

        