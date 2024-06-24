import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

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

class dataReadPip(object):

    def __init__(self, transforms=None, normalize=False, resize_width=None, enlarge=False, center_crop_size=None):

        self.normalize = normalize
        self.transforms = transforms
        self.resize_width = resize_width
        self.enlarge = enlarge
        self.center_crop_size = center_crop_size
    def __call__(self, item):

        img = cv2.imread(item[0])
        lab = cv2.imread(item[1], 0)   # lab is ground-truth label picture

        if len(lab.shape) != 2:
            lab = cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)
            
        ret, lab = cv2.threshold(lab, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        if self.resize_width is not None:
            img = cv2.resize(img, (self.resize_width, self.resize_width), fx=1, fy=1, interpolation=cv2.INTER_CUBIC if self.enlarge else cv2.INTER_AREA)
            lab = cv2.resize(lab, (self.resize_width, self.resize_width), fx=1, fy=1, interpolation=cv2.INTER_CUBIC if self.enlarge else cv2.INTER_AREA)
        
        if self.center_crop_size is not None:
            h, w, c = img.shape
            cx, cy = w // 2, h // 2
            width_half = self.center_crop_size//2
            img = img[cy-width_half:cy+width_half, cx-width_half:cx+width_half, :]
            lab = lab[cy-width_half:cy+width_half, cx-width_half:cx+width_half]
        
        if self.transforms is not None:
            img, lab = self.transforms(img, lab)

        img = _preprocess_img(img)
        lab = _preprocess_lab(lab)

        if self.normalize:
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))(img)
        return img, lab


def _preprocess_img(cvImage):
    '''
    :param cvImage: numpy HWC BGR 0~255
    :return: tensor img CHW BGR  float32 cpu 0~1
    '''

    cvImage = cvImage.transpose(2, 0, 1).astype(np.float32) / 255


    return torch.from_numpy(cvImage)

def _preprocess_lab(cvImage):
    '''
    :param cvImage: numpy 0(background) or 255(crack pixel)
    :return: tensor 0 or 1 float32
    '''
    cvImage = cvImage.astype(np.float32) / 255

    return torch.from_numpy(cvImage)


class loadedDataset(Dataset):
    """
    Create a torch Dataset from data
    """

    def __init__(self, imglist, preprocess=None):
        super(loadedDataset, self).__init__()
        self.imglist = imglist
        if preprocess is None:
            preprocess = lambda x: x
        self.preprocess = preprocess

    def __getitem__(self, index):
        return self.preprocess(self.imglist[index])

    def __len__(self):
        return len(self.imglist)


