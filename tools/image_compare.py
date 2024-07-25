import glob
import os
from tqdm import tqdm
import cv2
import numpy as np
from copy import deepcopy
import random

def readIndex(index_path, shuffle=False):
    img_list = []
    with open(index_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            item = lines.strip().split()
            img_list.append(item)
    file_to_read.close()
    if shuffle is True:
        random.shuffle(img_list)
    return img_list

def show_mask_on_image(img, mask):
    # _, img_bi = cv2.threshold(mask, 0, 256, cv2.THRESH_BINARY)
    img_bi = mask
    mask = cv2.normalize(mask, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    img_inx = img_bi ==  0
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    heatmap[img_inx] = 0
    heatmap = np.float32(heatmap) / 255
    a = 0.8
    cam = a*heatmap + (1-a)*np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def main(compare_dir, index_path, img_dataset, trans_heatmap=True):
    dirs = os.listdir(compare_dir)
    dirs.sort()
    dirs[0], dirs[1], dirs[2], dirs[3] = dirs[1], dirs[0], dirs[2], dirs[3]
    img_path_lst = readIndex(index_path)
    bar = tqdm(range(0, len(img_path_lst)), desc="concating images...", ncols=100)
    img_concat_dir = os.path.join("concatenated_imgs", img_dataset)
    if not os.path.exists(img_concat_dir):
        os.makedirs(img_concat_dir)
        
    for i in bar:
        img_origin_path, img_ann_path = img_path_lst[i][0], img_path_lst[i][1]
        img_origin = cv2.imread(img_origin_path)
        img_origin_border = cv2.copyMakeBorder(img_origin, 0, 0, 0, 20, cv2.BORDER_CONSTANT, value=[255,255,255])
        img_ann = cv2.imread(img_ann_path)
        h,w,c = img_ann.shape
        # ret, img_ann = cv2.threshold(img_ann, 0, 256, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        img_ann = cv2.copyMakeBorder(img_ann, 0, 0, 0, 20, cv2.BORDER_CONSTANT, value=[255,255,255])
        cv2.putText(img_ann, "GroundTruth", (30,30), cv2.FONT_HERSHEY_PLAIN, 2.0, (255,255,255) if trans_heatmap else (0,0,0), 2)
        img_concat = np.concatenate([img_origin_border, img_ann], axis=1)
        
        img_name = img_ann_path.split('/')[-1]
        for j in dirs:
            img_path = os.path.join(compare_dir, j, img_name)
            img_temp = cv2.imread(img_path, 0)
            
            if img_temp.shape != h:
                img_temp = cv2.resize(img_temp, dsize=(h,w), interpolation=cv2.INTER_LINEAR)
            
            if trans_heatmap:
                img_temp = show_mask_on_image(img_origin, img_temp)  # '/data/wc/CrackLab/compare_dir/crack500/SwinCrack/20160222_081011_1281_721.png'
            else:
                img_temp = ~img_temp
            # img_temp = (img_temp!=0)==(img_ann!=0)
            # img_sub = deepcopy(img_ann)
            # img_sub[img_temp]=0
            # img_temp = ~img_sub
            cv2.putText(img_temp, j, (30,30), cv2.FONT_HERSHEY_PLAIN, 2.0, (255,255,255) if trans_heatmap else (0,0,0), 2)
            img_temp = cv2.copyMakeBorder(img_temp, 0, 0, 0, 20, cv2.BORDER_CONSTANT, value=[255,255,255])
            img_concat = np.concatenate([img_concat, img_temp], axis=1)
        img_concat_name = img_name
        img_concat_path = os.path.join(img_concat_dir, img_concat_name)
        cv2.imwrite(img_concat_path, img_concat)
    bar.close()
    with open(os.path.join(img_concat_dir, "output.txt"),'a' , encoding='utf-8') as fout:
        line = ""
        for i in dirs:
            line += i + '                       '
        fout.write(line)

if __name__=="__main__":
    compare_dir = "compare_dir/Stone331"
    img_dataset = "DCNCrack_Stone331_val"
    dataset = "datasets/Stone331_val.txt"
    main(compare_dir, dataset, img_dataset, trans_heatmap=True)