import numpy as np
from tqdm import tqdm
import cv2
import torch
import os
import codecs
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from matplotlib.backends.backend_pdf import PdfPages
device = torch.device("cuda")


def preprocess_img_tensor(img_path, ann_path):
    '''
    :param pre: result predicted by model
    :param gt:  ground-truth label of the pre_img
    :return: image of prediction and ground-truth is torch tensor type
    '''
    pre_img = cv2.imread(img_path)
    gt_img  = cv2.imread(ann_path)
    if len(pre_img.shape) != 2:
        pre_img = cv2.cvtColor(pre_img, cv2.COLOR_BGR2GRAY)
    if len(gt_img.shape) != 2:
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
    if gt_img.shape[1] != pre_img.shape[1]:
        gt_img = cv2.resize(gt_img, pre_img.shape, interpolation=cv2.INTER_LINEAR)
    ret, gt_img = cv2.threshold(gt_img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    pre_img = ((gt_img == 0) + (gt_img == 255)) * pre_img
    pre_img = pre_img.astype(np.float32) / 255
    gt_img  = gt_img.astype(np.float32) / 255
    pre_img, gt_img = torch.from_numpy(pre_img), torch.from_numpy(gt_img)
    pre_img, gt_img = pre_img.type(torch.cuda.FloatTensor).to(device), gt_img.type(torch.cuda.FloatTensor).to(device)
    return pre_img, gt_img

def preprocess_img_list(img_path, ann_path):
    pre_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    gt_img  = cv2.imread(ann_path, cv2.IMREAD_UNCHANGED)
    # if len(pre_img.shape) != 2:
    #     pre_img = cv2.cvtColor(pre_img, cv2.COLOR_BGR2GRAY)
    if len(gt_img.shape) != 2:
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
        
    if gt_img.shape[1] != pre_img.shape[1]:
        gt_img = cv2.resize(gt_img, pre_img.shape, interpolation=cv2.INTER_LINEAR)
    # ret, gt_img = cv2.threshold(gt_img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # pre_img = ((gt_img == 0) + (gt_img == 255)) * pre_img
    # pre_img = pre_img.astype(np.float32)
    # gt_img  = gt_img.astype(np.float32)
    return pre_img, gt_img


def prediction_acc(pred, gt):
    '''
    :param pred: prediction image
    :param gt:   ground-truth label
    :return:     accuracy, precision, recall
    '''
    # accracy = pred.eq(gt.view_as(pred)).sum().item() / gt.numel()
    # precision = pred[gt > 0].eq(gt[gt > 0].view_as(pred[gt > 0])).sum().item() / (gt[gt > 0].numel() + 1e-6)
    tp = pred[gt > 0].eq(gt[gt > 0].view_as(pred[gt > 0])).sum().item()
    tn = pred[gt < 1].eq(gt[gt < 1].view_as(pred[gt < 1])).sum().item()
    fp = pred[gt < 1].eq(gt[gt < 1].view_as(pred[gt < 1])).numel() - pred[gt < 1].eq(gt[gt < 1].view_as(pred[gt < 1])).sum().item()
    fn = pred[gt > 0].eq(gt[gt > 0].view_as(pred[gt > 0])).numel() - pred[gt > 0].eq(gt[gt > 0].view_as(pred[gt > 0])).sum().item()
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    accuracy = (tp + tn) / gt.numel()
    return precision, recall, accuracy

def plt_curve(prlist:list, method, legend, output):
    '''
    :param prlist:    recall_ary, precision_ary, ap
    :param precision_ary: array of the precision results
    :return: plot the precision-recall curve
    '''
    _, axs = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    legend_value = prlist[2]
    axs.plot(prlist[0], prlist[1], '-', label='[{}={:.03f}]{}'.format(legend, legend_value, method), lw=2)
    font1 = {
        'weight': 'normal',
        'size': 14, }
    axs.grid(True, linestyle='-.')
    axs.set_xlim([0., 1.])
    axs.set_ylim([0., 1.])
    axs.set_title('Precision Recall Curve', font1)
    axs.set_xlabel("Recall", font1)
    axs.set_ylabel("Precision", font1)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    axs.legend(loc="lower left")
    pdf = PdfPages(r'{}'.format(output))
    plt.savefig(pdf, format='pdf', bbox_inches='tight')
    pdf.close()
    pdf=None
    #plt.show()


def get_statistics(pred, gt):
    """
    return tp, fp, fn
    """
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    kernel = np.ones((5, 5), np.uint8)
    gtd = cv2.dilate(gt, kernel, iterations = 1)
    tp = np.sum((pred == 1) & (gtd == 1))

    return [tp, fp, fn]


def get_statistic_list(pred, gt, num_cls=2):
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

# 计算 ODS 方法
def cal_ods_metrics(pred_list, gt_list, thresh_step=0.01):
    final_accuracy_all = []
    p_bar = tqdm(np.arange(0.0, 1.0, thresh_step), desc="calculate ODS: ", ncols=100)
    for thresh in p_bar:
        statistics = []
        for pred, gt in zip(pred_list, gt_list):
            gt_img = (gt / 255).astype('uint8')
            pred_img = ((pred / 255) > thresh).astype('uint8')
            # calculate each image
            statistics.append(get_statistics(pred_img, gt_img))
        # get tp, fp, fn
        tp = np.sum([v[0] for v in statistics])
        fp = np.sum([v[1] for v in statistics])
        fn = np.sum([v[2] for v in statistics])
        # calculate precision
        p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
        # calculate recall
        r_acc = tp / (tp + fn + 1e-5)
        # calculate f-score
        final_accuracy_all.append([thresh, p_acc, r_acc, 2 * p_acc * r_acc / (p_acc + r_acc)])
    p_bar.close()
    return final_accuracy_all

# 计算 OIS 方法
def cal_ois_metrics(pred_list, gt_list, thresh_step=0.01):
    final_acc_all = []
    p_bar = tqdm(range(len(gt_list)), desc="calculate OIS: ", ncols=100)
    for i in p_bar:
        pred, gt = pred_list[i], gt_list[i]
        statistics = []
        for thresh in np.arange(0.0, 1.0, thresh_step):
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred / 255 > thresh).astype('uint8')
            tp, fp, fn = get_statistics(pred_img, gt_img)
            p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
            r_acc = tp / (tp + fn + 1e-6)
            if p_acc + r_acc == 0:
                f1 = 0
            else:
                f1 = 2 * p_acc * r_acc / (p_acc + r_acc)
            statistics.append([thresh, f1])
        max_f = np.amax(statistics, axis=0)
        final_acc_all.append(max_f[1])
    p_bar.close()
    return np.mean(final_acc_all)

def cal_global_acc(pred, gt):
    """
    acc = (TP+TN)/all_pixels
    """
    h,w = gt.shape
    return [np.sum(pred==gt), float(h*w)]

# 计算 Semantic metrics: Global Accuracy，Mean Accuracy 和 Mean Iou：
def cal_semantic_metrics(pred_list, gt_list, thresh_step=0.01, num_cls=2):
    final_accuracy_all = []
    p_bar = tqdm(np.arange(0.0, 1.0, thresh_step), desc="calculate semantic metrics: ", ncols=100)
    for thresh in p_bar:
        global_accuracy_cur = []
        statistics = []
        for pred, gt in zip(pred_list, gt_list):
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred / 255 > thresh).astype('uint8')
            # calculate each image
            global_accuracy_cur.append(cal_global_acc(pred_img, gt_img))
            statistics.append(get_statistic_list(pred_img, gt_img, num_cls))
        # get global accuracy with corresponding threshold: (TP+TN)/all_pixels
        global_acc = np.sum([v[0] for v in global_accuracy_cur]) / np.sum([v[1] for v in global_accuracy_cur])
        # get tp, fp, fn
        counts = []
        for i in range(num_cls):
            tp = np.sum([v[i][0] for v in statistics])
            fp = np.sum([v[i][1] for v in statistics])
            fn = np.sum([v[i][2] for v in statistics])
            counts.append([tp, fp, fn])

        # calculate mean accuracy
        mean_acc = np.sum([v[0] / (v[0] + v[2]) for v in counts]) / num_cls
        # calculate mean iou
        mean_iou_acc = np.sum([v[0] / (np.sum(v)) for v in counts]) / num_cls
        final_accuracy_all.append([thresh, global_acc, mean_acc, mean_iou_acc])
    p_bar.close()
    return final_accuracy_all

def save_results(input_list, output_path):
    with codecs.open(output_path, 'w', encoding='utf-8') as fout:
        for ll in input_list:
            line = '\t'.join(['%.4f'%v for v in ll])+'\n'
            fout.write(line)

def load_img(pre_dir, ann_dir, pre_img_list, pre_suffix, ann_suffix):
    pre_tensor=None
    ann_tensor=None
    img_list = []
    ann_list = []
    p_bar = tqdm(range(0, len(pre_img_list)), desc="load images: ", ncols=100)
    for i in p_bar:
        pre_path = os.path.join(pre_dir, pre_img_list[i])
        ann_path = os.path.join(ann_dir, pre_img_list[i].split('.')[0]+ann_suffix)
        if i == 0:
            pre_tensor, ann_tensor = preprocess_img_tensor(pre_path, ann_path)
            continue
        pre_t, gt_t = preprocess_img_tensor(pre_path, ann_path)
        pre_tensor = torch.cat((pre_tensor, pre_t), 0)
        ann_tensor = torch.cat((ann_tensor, gt_t), 0)
        img, ann = preprocess_img_list(pre_path, ann_path)
        img_list.append(img)
        ann_list.append(ann)
    p_bar.close()
    return pre_tensor, ann_tensor, img_list, ann_list

def archive_evaluation(method, dataset, ap, ods, ois, max_fa, output):
    with open(output, 'a', encoding='utf-8') as fout:
        line = "Method:{}|dataset:{}|AP:{:.6f} | ODS:{:.6f} | OIS:{:.6f} | Global Accuracy:{:.6f} | Class Average Accuracy:{:6f} | Mean IOU:{:6f}".format(method, dataset, ap, ods, ois, max_fa[1], max_fa[2], max_fa[3]) + '\n'
        fout.write(line)

def archive_pr_sklearn(output, precision, recall, threshold):
    with codecs.open(output, 'w', encoding='utf-8') as fout:
        for i in range(0, len(precision)-1):
            ll = [precision[i], recall[i], threshold[i]]
            line = '\t'.join(['%.4f'%v for v in ll])+'\n'
            fout.write(line)
            
def plot_prc_ods(output, fname, final_ods):
    _, axs = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
    
    p_acc = [k[1] for k in final_ods]
    r_acc = [k[2] for k in final_ods]
    f_acc = [k[3] for k in final_ods]
    
    max_index = np.argmax(np.array(f_acc))
    ap = np.trapz(p_acc, r_acc)
    axs.plot(np.array(r_acc), np.array(p_acc), label='[AP={:.03f}]{}'.format(ap, fname).replace('=0.', '=.'), lw=2)

    axs.grid(True, linestyle='-.')
    axs.set_xlim([0., 1.])
    axs.set_ylim([0., 1.])
    axs.set_xlabel('{}'.format("recall"))
    axs.set_ylabel('{}'.format("precision"))
    axs.legend(loc='{}'.format("lower left"))

    pdf = PdfPages(r'{}'.format(output))
    plt.savefig(pdf, format='pdf', bbox_inches='tight')
    pdf.close()
    pdf=None
    
if __name__ == '__main__':
    # --------需要修改的参数---------
    pre_dir = '/data/wc/CrackLab/inference/v100/inference/DCNCrack_abl/DCNCrack-CAMFI'
    ann_dir = '/data/wc/Dataset/DeepCrackTP_datasets/CrackLS315/ann/'
    pre_suffix = '.bmp'
    ann_suffix = '.bmp'
    dataset = "CrackLS315"
    method = pre_dir.split('/')[-1]
    legend = "AP"  # Metrics displayed on the curve fig: "OIS", "ODS", and "AP"
    # -----------------------------
    output_dir = os.path.join("evaluation", dataset, method)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pre_img_list = os.listdir(pre_dir)
    pre_img_list.sort()
    # ann_img_list = os.listdir(ann_dir)
    # ann_img_list.sort()

    pre_tensor, ann_tensor, img_list, ann_list = load_img(pre_dir, ann_dir, pre_img_list, pre_suffix, ann_suffix)
    final_ods = cal_ods_metrics(img_list, ann_list, 0.01)
    final_ois = cal_ois_metrics(img_list, ann_list, 0.01)
    final_accuracy_all = cal_semantic_metrics(img_list, ann_list)

    max_f = np.amax(final_ods, axis=0)
    thresh_sm = max_f[0]
    ods = max_f[3]
    ois = final_ois
    max_fa = np.amax(final_accuracy_all, axis=0)
    pre_tensor, ann_tensor = torch.flatten(pre_tensor), torch.flatten(ann_tensor)
    precision_np = pre_tensor.cpu().numpy()
    gt_np = ann_tensor.cpu().numpy()
    precision, recall, threshold = precision_recall_curve(gt_np, precision_np)
    ap = average_precision_score(gt_np, precision_np)
    print("Average precision is:", ap)
    print("------------Sensitivity and Specificity Metrics---------------\n")
    print("OIS:", ois)
    print("ODS:", ods)
    print("optim thresh by Specificity Metrics:", thresh_sm)
    print("-----------Segmentation Metrics---------------\n")
    print("Global Accuracy:", max_fa[1])
    print("Class Average Accuracy:", max_fa[2])
    print("Mean IOU:", max_fa[3])
    print("optim thresh by Segmentation Metrics:", max_fa[0])
    
    # plot curve using the skilearn module
    output = method + '_' + dataset +"_sklearn_curve.pdf"
    output = os.path.join(output_dir, output)
    if legend == "AP":
        plt_curve([recall, precision, ap], method, legend, output)
    elif legend == "OIS":
        plt_curve([recall, precision, ois], method, legend, output)
    elif legend == "ODS":
        plt_curve([recall, precision, ods], method, legend, output)

    # archive the evaluation results of the model
    output = method + '_' + dataset + "_evaluation" + '.txt'
    output = os.path.join(output_dir, output)
    archive_evaluation(method, dataset, ap, ods, ois, max_fa, output)

    # archive the precisoin-recall data by sklearn    
    output = method + '_' + dataset + "_sklearn" + '.txt'
    output = os.path.join(output_dir, output)
    archive_pr_sklearn(output, precision, recall, threshold)
            
    # archive the precision-recall data by ods cal
    output = method + '_' + dataset + "_ods" + '.prf'
    output = os.path.join(output_dir, output)
    save_results(final_ods, output)
    
    # plot curve using the results from ods calculation
    output = method + '_' + dataset + "_ods_curve.pdf"
    output = os.path.join(output_dir, output)
    fname = dataset + method
    plot_prc_ods(output, fname, final_ods)










