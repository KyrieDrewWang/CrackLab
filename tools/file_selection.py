import os
import shutil

src_dir = "compare_dir/crack500/DeepCrack"
tar_dir = "/data/wc/Dataset/DeepCrackTP_datasets/crack500/img"
save_dir = "compare_dir/crack500/Input"
dir_list = os.listdir(src_dir)
tar_lists = os.listdir(tar_dir)
for d in dir_list:
    d = d.split('.')[0] + '.jpg'
    if d in tar_lists:
        src_path = os.path.join(tar_dir, d)
        shutil.move(src_path, save_dir)
    