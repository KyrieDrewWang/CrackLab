import codecs
import os
import sys

import cv2
import numpy as np

sys.path.append("/data/wc/CrackLab/")
from data.dataset import dataReadPip, loadedDataset, readIndex


def imread(path, load_size=0, load_mode=cv2.IMREAD_GRAYSCALE, convert_rgb=False, thresh=-1):
    im = cv2.imread(path, load_mode)
    if convert_rgb:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if load_size > 0:
        im = cv2.resize(im, (load_size, load_size), interpolation=cv2.INTER_CUBIC)
    if thresh > 0:
        _, im = cv2.threshold(im, thresh, 255, cv2.THRESH_BINARY)
    return im

def get_img_pairs(pred_img_dir, ann_img_dir):
    pred_img_paths = os.listdir(pred_img_dir)
    ann_img_paths  = os.listdir(ann_img_dir)
    pred_img_paths.sort()
    ann_img_paths.sort()
    assert len(pred_img_paths)==len(ann_img_paths)
    pred_img_list, ann_img_list =[],[]
    for i in range(len(pred_img_paths)):
        pred_path = os.path.join(pred_img_dir, pred_img_paths[i])
        ann_path  = os.path.join(ann_img_dir, ann_img_paths[i])
        pred_img_list.append(imread(pred_path))
        ann_img_list.append(imread(ann_path))
    return pred_img_list, ann_img_list

def save_results(input_list, output_path):
    with codecs.open(output_path, 'w', encoding='utf-8') as fout:
        for ll in input_list:
            line = '\t'.join(['%.4f'%v for v in ll])+'\n'
            fout.write(line)


def cal_prf_metrics(pred_list, gt_list, thresh_step=0.01):
    """
    Calculate sensitivity and specificity metrics:
    - Precision
    - Recall
    - F-score
    """

    final_accuracy_all = []

    for thresh in np.arange(0.0, 1.0, thresh_step):
        print(thresh)
        statistics = []
        
        for pred, gt in zip(pred_list, gt_list):
            gt_img   = (gt/255).astype('uint8')
            pred_img = (pred/255 > thresh).astype('uint8')
            # calculate each image
            statistics.append(get_statistics(pred_img, gt_img))
        
        # get tp, fp, fn
        tp = np.sum([v[0] for v in statistics])
        fp = np.sum([v[1] for v in statistics])
        fn = np.sum([v[2] for v in statistics])

        # calculate precision
        p_acc = 1.0 if tp==0 and fp==0 else tp/(tp+fp)
        # calculate recall
        r_acc = tp/(tp+fn)
        # calculate f-score
        final_accuracy_all.append([thresh, p_acc, r_acc, 2*p_acc*r_acc/(p_acc+r_acc)])
    return final_accuracy_all

def get_statistics(pred, gt):
    """
    return tp, fp, fn
    """
    tp = np.sum((pred==1)&(gt==1))
    fp = np.sum((pred==1)&(gt==0))
    fn = np.sum((pred==0)&(gt==1))
    return [tp, fp, fn] 

def cal_semantic_metrics(pred_list, gt_list, thresh_step=0.01, num_cls=2):
    """
    Calculate Segmentation metrics:
    - GlobalAccuracy
    - MeanAccuracy
    - Mean MeanIoU
    """
    final_accuracy_all = []
    
    for thresh in np.arange(0.0, 1.0, thresh_step):
        print(thresh)
        global_accuracy_cur = []
        statistics = []
        for pred, gt in zip(pred_list, gt_list):
            gt_img   = (gt/255).astype('uint8')
            pred_img = (pred/255 > thresh).astype('uint8')
            # calculate each image
            global_accuracy_cur.append(cal_global_acc(pred_img, gt_img))
            statistics.append(get_statistics_cls(pred_img, gt_img, num_cls))
        
        # get global accuracy with corresponding threshold: (TP+TN)/all_pixels
        global_acc = np.sum([v[0] for v in global_accuracy_cur])/np.sum([v[1] for v in global_accuracy_cur])
        
        # get tp, fp, fn
        counts = []
        for i in range(num_cls):
            tp = np.sum([v[i][0] for v in statistics])
            fp = np.sum([v[i][1] for v in statistics])
            fn = np.sum([v[i][2] for v in statistics])
            counts.append([tp, fp, fn])

        # calculate mean accuracy
        mean_acc = np.sum([v[0]/(v[0]+v[2]) for v in counts])/num_cls
        # calculate mean iou
        mean_iou_acc = np.sum([v[0]/(np.sum(v)) for v in counts])/num_cls
        final_accuracy_all.append([thresh, global_acc, mean_acc, mean_iou_acc])

    return final_accuracy_all

def cal_global_acc(pred, gt):
    """
    acc = (TP+TN)/all_pixels
    """
    h,w = gt.shape
    return [np.sum(pred==gt), float(h*w)]

def get_statistics_cls(pred, gt, num_cls=2):
    """
    return tp, fp, fn
    """
    h,w = gt.shape
    statistics = []
    for i in range(num_cls):
        tp = np.sum((pred==i)&(gt==i))
        fp = np.sum((pred==i)&(gt!=i))
        fn = np.sum((pred!=i)&(gt==i)) 
        statistics.append([tp, fp, fn])
    return statistics


if __name__ == "__main__":
    pred_dir = "deepcrack/CrackTree260-DeepCrack/"
    ann_dir  = "DeepCrackTP_datasets/CrackTree260/CrackTree260_512/CrackTree260_ann/test"
    metric_mode = 'prf'
    method_name = os.path.split(os.path.split(pred_dir)[0])[1].split("_")[0]
    output = os.path.join("plot", method_name+".prf")
    pred_list, ann_list = get_img_pairs(pred_dir, ann_dir)
    final_results = []
    if metric_mode == 'prf':
        final_results = cal_prf_metrics(pred_list=pred_list, gt_list=ann_list, thresh_step=0.01)
    elif metric_mode == 'sem':
        final_results = cal_semantic_metrics(pred_list=pred_list, gt_list=ann_list, thresh_step=0.01)
    else:
        print("unknow mode of metrics")
    save_results(final_results, output)
