import sys
sys.path.append('config')
sys.path.append('.')
import cv2
import numpy as np
import torch
import os
from nets.crackformer import crackformer
from config.config_crackformer import Config as cfg
from torchvision import transforms
from torchvision.transforms import Compose, Normalize, ToTensor
from data.dataset import readIndex
from tqdm import tqdm

def preprocess_image(img: np.ndarray, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) -> torch.Tensor:
    preprocessing = Compose([
        ToTensor(),
        # Normalize(mean=mean, std=std)
    ])
    return preprocessing(img.copy()).unsqueeze(0)


class SaveHook:
    def __init__(self) -> None:
        super().__init__()
        self.out = []   
             
    def __call__(self, module, module_in, module_out):
        self.out.append(module_out)
                
    def clear(self):
        self.out = []


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
    
    trained_model = "checkpoints/CrackLS315-CrackFormer/CrackLS315-CrackFormer_epoch(103)_acc(0.65945_0.65945)_0000031_2023-03-21-12-45-26.pth"
    base_dir = "visual/visualization/crackformer_attn_vis"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id
    device = torch.device("cuda")
    img_ann_list = readIndex(cfg.test_data_path)
    
    model = crackformer()
    model.load_state_dict(torch.load(trained_model))
    model = model.cuda()
    model.eval()
    bar = tqdm(img_ann_list, desc="visualizing offset:", ncols=100, unit="fps")
    for img_path, ann_path in bar:
        
        img_name = img_path.split('/')[-1].split('.')[0]
        img_dir = os.path.join(base_dir, img_name)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        
        rgb_img = cv2.imread(img_path)
        input_tensor = np.float32(rgb_img) / 255
        input_tensor = preprocess_image(input_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        input_tensor = input_tensor.type(torch.cuda.FloatTensor).to(device)
        savehook = SaveHook()
        hk1 = model.LABlock_1.register_forward_hook(savehook)
        hk2 = model.LABlock_2.register_forward_hook(savehook)
        hk3 = model.LABlock_3.register_forward_hook(savehook)
        hk4 = model.LABlock_4.register_forward_hook(savehook)
        hk5 = model.LABlock_5.register_forward_hook(savehook)
        output = model(input_tensor)
        hk1.remove()
        hk2.remove()
        hk3.remove()
        hk4.remove()
        hk5.remove()
        discard_ratio = 0.
        for inx, attn in enumerate(savehook.out):
            # attn = torch.mean(attn, dim=1)
            attn = torch.max(attn, axis=1)[0]
            flat_attn = attn.view(attn.size(0), -1)
            _, indices = flat_attn.topk(int(flat_attn.size(-1)*discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat_attn[:, indices] = 1E-5
            attn = attn / attn.sum(dim=-1)
            attn = attn.squeeze(0).cpu().detach().numpy()
            attn = cv2.resize(attn, (512, 512), interpolation=cv2.INTER_CUBIC)
            attn = (attn - np.min(attn)) / (np.max(attn) - np.min(attn))
            i_m = show_mask_on_image(rgb_img, attn)
            img_path = os.path.join(img_dir, str(inx)+"_heatmap.png")
            cv2.imwrite(img_path, i_m)
        savehook.clear()
    bar.close()
