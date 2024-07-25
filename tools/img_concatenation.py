import cv2
import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

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

def compare(i, added_compare_dir, ground_truth_dir, concat_img_dir):

    input_path = os.path.join(input_dir, i.replace('.png', '.jpg'))
    img_origin = cv2.imread(input_path)
    input_img  = cv2.copyMakeBorder(img_origin, 0, 0, 0, 20, cv2.BORDER_CONSTANT, value=[255,255,255])
    gt_img_path = os.path.join(ground_truth_dir, i)#.split('-')[1])
    gt_img = cv2.imread(gt_img_path)
    gt_img = cv2.copyMakeBorder(gt_img, 0, 0, 0, 20, cv2.BORDER_CONSTANT, value=[255,255,255])
    cv2.putText(gt_img, "GroundTruth", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.,(255,255,255), 3, cv2.LINE_AA)
    img_concat = np.hstack([input_img, gt_img])
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
        added_compare_img = show_mask_on_image(img_origin, added_compare_img)
        added_compare_img = cv2.copyMakeBorder(added_compare_img, 0, 0, 0, 20, cv2.BORDER_CONSTANT, value=[255,255,255])
        cv2.putText(added_compare_img, desc, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1., (255,255,255), 3, cv2.LINE_AA )
        img_concat = np.hstack([img_concat, added_compare_img])
    img_concat_path = os.path.join(concat_img_dir, i)
    cv2.imwrite(img_concat_path, img_concat)

if __name__ == "__main__":
    ground_truth_dir = "compare_dir/crack500/GT"
    input_dir = "compare_dir/crack500/Input"
    added_compare_dir= {
        "DCNCrack":     "compare_dir/crack500/DMANet",
        "CrackFormer":  "compare_dir/crack500/CrackFormer",
        "DeepCrack":    "compare_dir/crack500/DeepCrack",
        "Unet":         "compare_dir/crack500/Unet",
        "SwinUnet":     "compare_dir/crack500/SwinCrack",
        "SegNet":       "compare_dir/crack500/SegNet",
        "SDDNet":       "compare_dir/crack500/CrackDecoder",
    }
    prefix = 'png'
    concat_img_dir = list(added_compare_dir.values())[0] + "_concat"
    if not os.path.exists(concat_img_dir):
        os.makedirs(concat_img_dir)
    fnames = os.listdir(ground_truth_dir)
    g_img_names=[]
    for f in fnames:
        if f.endswith(prefix):
            g_img_names.append(f)
        else:
            continue
    po = Pool(processes=32)
    for i in g_img_names:
        po.apply_async(compare, args=(i, added_compare_dir, ground_truth_dir, concat_img_dir,))
        # compare(i, added_compare_dir, ground_truth_dir, concat_img_dir,)
    po.close()
    po.join()