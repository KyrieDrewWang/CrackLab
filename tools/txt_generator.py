import os
import random


dataset_type = ""  # train or test
base_dir = '/data/Dataset/DeepCrackTP_datasets/crack500' # absolute diretory of the dataset
image_dir = "img"  # dirname of the image dir
ann_dir   = "ann"  # dirname of the mask dir
dataset_name = "crack500" # dataset name
img_suffix = '.jpg' # suffix of the image file
ann_suffix = '.png' # suffix of the mask file
num = 1896
train_image_dir = os.path.join(base_dir, image_dir)
train_mask_dir  = os.path.join(base_dir, ann_dir)
imageList=[]
maskList=[]
train_image_fnames = os.listdir(train_image_dir)
# train_image_fnames.sort(key=lambda x: (int(x[len(dataset_name)+1:-len(img_suffix)])))
# train_image_fnames.sort(key=lambda x: (int(x[:-len(img_suffix)])))
# for name in random.sample(train_image_fnames, num):  # len(train_image_fnames)-195):
for name in train_image_fnames:
    img_path = os.path.join(train_image_dir, name)
    if img_path in imageList:
        continue
    mask_path = os.path.join(train_mask_dir, name.replace(img_suffix, ann_suffix))
    if not os.path.exists(mask_path):
        continue
    imageList.append(img_path)
    maskList.append(mask_path)

text_file_path = os.path.join("rtx_data_text", dataset_name+"_"+dataset_type+".txt")
textfile = open(text_file_path, "w")
for im,mas in zip(imageList, maskList):
    textfile.write(im+' '+mas + "\n")
textfile.close()