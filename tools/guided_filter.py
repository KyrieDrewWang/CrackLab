import os
import sys
sys.path.append('.')
sys.path.append('/data/wc/CrackLab/')
sys.path.append('config')
import cv2
import numpy as np
from cv2.ximgproc import guidedFilter
from config.config_deepseg import Config as cfg
if __name__ == '__main__':
    radius = cfg.radius
    eps = epsilon = cfg.epsilon # help='eps = 1e-6*img_siz*img_siz'
    desc = 'deepseg'
    inference_path = "compare_dir/WHCF218"
    gf_dir = os.path.join(inference_path, desc, "pred")

    if not os.path.exists(gf_dir):
        os.makedirs(gf_dir)

    # pred_path = os.path.join(inference_path, desc, "pred")
    pred_path = os.path.join(inference_path, desc, "gff")
    fuse_path = os.path.join(inference_path, desc, "fuse5")

    pred_path_list = os.listdir(pred_path)
    fuse_path_list = os.listdir(fuse_path)
    pred_path_list.sort()
    fuse_path_list.sort()

    for i in range(0, len(pred_path_list)):
        img_path = os.path.join(pred_path, pred_path_list[i])
        fus_path = os.path.join(fuse_path, fuse_path_list[i])
        
        pred_img = (cv2.imread(img_path, cv2.IMREAD_UNCHANGED) > 0.31*255).astype('uint8')*255
        gf_img   = cv2.imread(fus_path, cv2.IMREAD_UNCHANGED)
        img_filtered = guidedFilter(gf_img, pred_img, radius, eps)
        img_filtered_path = os.path.join(gf_dir, pred_path_list[i])
        cv2.imwrite(img_filtered_path, img_filtered, [cv2.IMWRITE_PNG_COMPRESSION, 0])