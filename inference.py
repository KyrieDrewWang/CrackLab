import os
import sys
sys.path.append('.')
sys.path.append('./config')
sys.path.append('./nets')
import cv2
import numpy as np
import torch
from tqdm import tqdm
from data.dataset import dataReadPip, loadedDataset, readIndex
def test(net_name, model_path):
    config_name = 'config_' + net_name
    config = __import__(config_name)
    cfg = getattr(config, "Config")()
    network = __import__(cfg.network_name)
    save_path = os.path.join('.', 'inference', cfg.name.split('-')[0], ''.join(cfg.name.split('-')[1:]))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if net_name == "deepseg":
        fuse5_path = os.path.join('.', 'inference', cfg.name.split('-')[0], cfg.network_name, "fuse5")
        if not os.path.exists(fuse5_path):
            os.mkdir(fuse5_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id
    
    test_pipline = dataReadPip(transforms=None, resize_width=cfg.resize_width, enlarge=cfg.enlarge, center_crop_size=cfg.center_crop_size)
    test_list = readIndex(cfg.test_data_path)
    test_dataset = loadedDataset(test_list, preprocess=test_pipline)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.test_batch_size,shuffle=False, num_workers=4, drop_last=False)
    # -------------------- build trainer --------------------- #
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    num_gpu = torch.cuda.device_count()
    model = getattr(network, cfg.network_name)()
    model.to(device)
    model_pth = torch.load(model_path, map_location=device)
    model.load_state_dict(model_pth, strict=True)
    model.eval()
    with torch.no_grad():
        for names, (img, lab) in tqdm(zip(test_list, test_loader)):
            test_data, test_target = img.type(torch.cuda.FloatTensor).to(device), lab.type(torch.cuda.FloatTensor).to(
                device)
            test_pred = model(test_data)
            test_pred = torch.sigmoid(test_pred[0].cpu().squeeze())
            save_name = os.path.join(save_path, os.path.split(names[1])[1])           
            save_pred = torch.zeros((cfg.IMG_SIZE, cfg.IMG_SIZE))
            save_pred[:cfg.IMG_SIZE, :cfg.IMG_SIZE] = test_pred
            save_pred = save_pred.numpy() * 255
            cv2.imwrite(save_name, save_pred.astype(np.uint8))
            if net_name == "deepseg":
                guided_pred = torch.sigmoid(test_pred[1].cpu().squeeze())
                g=os.path.split(names[1])[1].split('.')
                fuse5_name = os.path.join(fuse5_path, g[0] + '_fuse1.' + g[1])
                guid_pred = torch.zeros((cfg.IMG_SIZE, cfg.IMG_SIZE))
                guid_pred[:cfg.IMG_SIZE, :cfg.IMG_SIZE] = guided_pred
                guid_pred = guid_pred.numpy() * 255
                cv2.imwrite(fuse5_name, guid_pred.astype(np.uint8))
if __name__ == '__main__':
    net_name = "crackformer"
    model_path = 'checkpoints/compairision_methods/Crack500/crack500-CrackFormer/crack500-CrackFormer_epoch(39)_acc(0.83949_0.82532)_0000050_2023-09-19-16-38-21.pth'
    test(net_name, model_path)