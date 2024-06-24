import cv2
import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

def compare(i, added_compare_dir, ground_truth_dir, concat_img_dir):
    gt_img_path = os.path.join(ground_truth_dir, i)#.split('-')[1])
    gt_img = cv2.imread(gt_img_path)
    cv2.putText(gt_img, "src", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.,(255,255,255), 3, cv2.LINE_AA)
    img_concat = gt_img
    for desc in added_compare_dir.keys():  
        added_compare_path=added_compare_dir[desc]
        try:
            added_compare_img_path = os.path.join(added_compare_path, i)
            added_compare_img = cv2.imread(added_compare_img_path)
        except Exception as e:
            print("img {} does not exist in the {}".format(i, added_compare_path))
            return
        if added_compare_img is None:
            added_compare_img = np.ones_like(gt_img)
        cv2.putText(added_compare_img, desc, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1., (255,255,255), 3, cv2.LINE_AA )
        img_concat = np.hstack([img_concat, added_compare_img])
    img_concat_path = os.path.join(concat_img_dir, i)
    cv2.imwrite(img_concat_path, img_concat)

if __name__ == "__main__":
    ground_truth_dir = "/data/wc/CrackLab/compare_dir/CrackLS315/GT"
    added_compare_dir= {
        "DCNCrack":     "/data/wc/CrackLab/compare_dir/CrackLS315/DCNCrack",
        "CrackFormer":  "/data/wc/CrackLab/compare_dir/CrackLS315/CrackFormer",
        "DeepCrack":    "/data/wc/CrackLab/compare_dir/CrackLS315/DeepCrack",
        "SDDNet":       "/data/wc/CrackLab/compare_dir/CrackLS315/SDDNet",
        "SegNet":       "/data/wc/CrackLab/compare_dir/CrackLS315/SegNet",
        "SwinUnet":     "/data/wc/CrackLab/compare_dir/CrackLS315/SwinUnet",
        "Unet":         "/data/wc/CrackLab/compare_dir/CrackLS315/Unet"
    }
    concat_img_dir = list(added_compare_dir.values())[0] + "_concat"
    if not os.path.exists(concat_img_dir):
        os.makedirs(concat_img_dir)
    fnames = os.listdir(ground_truth_dir)
    g_img_names=[]
    for f in fnames:
        if f.endswith('.bmp'):
            g_img_names.append(f)
        else:
            continue
    po = Pool(processes=32)
    for i in g_img_names:
        po.apply_async(compare, args=(i, added_compare_dir, ground_truth_dir, concat_img_dir,))
    po.close()
    po.join()