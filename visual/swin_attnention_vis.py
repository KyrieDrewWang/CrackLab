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
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from torchvision.transforms import Compose, Normalize, ToTensor, transforms
import math
from tqdm import tqdm
from data.dataset import readIndex

def preprocess_image(img: np.ndarray, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) -> torch.Tensor:
    preprocessing = Compose([
        ToTensor(),
        # Normalize(mean=mean, std=std)
    ])
    return preprocessing(img.copy()).unsqueeze(0)


def reshape_transform(tensor):
    b, l, c = tensor.shape
    height, width = int(math.sqrt(l)), int(math.sqrt(l))
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

if __name__ == '__main__':
    

    trained_model = "checkpoints/CrackLS315-SwinCrack-base/CrackLS315-SwinCrack-base_epoch(135)_acc(0.61709_0.61709)_0000028_2023-04-27-17-55-46.pth"
    base_dir = "visual/visualization/CrackLS315-SwinCrack-base"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id
    device = torch.device("cuda")
    img_ann_list = readIndex(cfg.test_data_path)
    model = swincrack()
    model.load_state_dict(torch.load(trained_model, map_location=device), strict=True)
    model = model.to(device=device)
    model.eval()
    bar = tqdm(img_ann_list, desc="visualizing offset:", ncols=100, unit="fps")
    for img_path, ann_path in bar:
        torch.cuda.empty_cache() 
        img_name = img_path.split('/')[-1].split('.')[0]
        img_dir = os.path.join(base_dir, img_name)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        rgb_img = cv2.imread(img_path)
        rgb_img = np.float32(rgb_img) / 255
        input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        input_tensor = input_tensor.type(torch.cuda.FloatTensor).to(device)
        
        for inx in range(1, len(cfg.DEPTHS)):
            # target_layer = [model.swin_unet.layers_up[inx].blocks[-1].identity]
            target_layer = [model.swin_unet.layers_up[inx].blocks[-1].norm2]
            cam = GradCAM(
                model=model,
                target_layers=target_layer,
                use_cuda=True,
                reshape_transform=reshape_transform
            )
            # cam = AblationCAM(
            #     model=model,
            #     target_layers=target_layer,
            #     use_cuda=True,
            #     reshape_transform=reshape_transform,
            #     ablation_layer=AblationLayerVit()
            # )
            cam.batch_size = 1 
            grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=None,
                eigen_smooth=False,
                aug_smooth=False
            )
            grayscale_cam = grayscale_cam[0, :]
            cam_image = show_cam_on_image(rgb_img, grayscale_cam)
            img_path = os.path.join(img_dir, str(inx)+"_cam.jpg")
            cv2.imwrite(img_path, cam_image)
    bar.close()