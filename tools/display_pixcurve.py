import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import os
from scipy.interpolate import make_interp_spline


def plot_curve(x, lim, curs:list, f_names:list, output:str):
    _, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    x_smooth = np.linspace(x.min(), x.max(), 1024)  # np.linspace 等差数列,从x.min()到x.max()生成300个数，便于后续插值
    for inx, i in enumerate(curs):
        i_smooth = make_interp_spline(x, i)(x_smooth)
        axs.plot(x_smooth, np.array(i_smooth), label='{}'.format(f_names[inx]), lw=2)
    axs.grid(True, linestyle='-.')
    axs.set_xlim([lim[0], lim[1]])
    axs.set_ylim([0, 350])
    axs.set_xlabel('{}'.format("Position along x direction"))
    axs.set_ylabel('{}'.format("Probability value"))
    axs.legend(loc='{}'.format("upper right"))
    
    pdf = PdfPages(r'{}'.format(output))
    plt.savefig(pdf, format='pdf', bbox_inches='tight')
    pdf.close()
    pdf=None

def pixel_inspect(img_path, lim:int, pos:int):
    img = cv2.imread(img_path, 0)
    _, col= img.shape
    cur = []
    for i in range(lim[0], min(col, lim[1])):
        cur.append(img[pos, i])
    return cur

def main(home_dir, img_name, lim, pos, output):
    img_dirs = os.listdir(home_dir)
    curs = []
    f_names = []
    for i in img_dirs:
        img_path = os.path.join(home_dir, i, img_name)
        cur = pixel_inspect(img_path, lim, pos)
        curs.append(cur)
        f_names.append(i)
    plot_curve(np.arange(lim[0], min(512, lim[1])), lim, curs, f_names, output)
        
if __name__ == '__main__':
    
    home_dir = "compare_dir/Stone331"
    dataset = "Stone331"
    img_name = "302.bmp"
    lim = (145, 155)
    pos = 210
    log = True
    
    output = dataset + "_" + img_name.split('.')[0] + ".pdf"
    main(home_dir, img_name, lim, pos, output)
    if log:
        with open("pixel_value_curve_info_dcn.txt", 'a') as f:
                f.write('dataset:%s'%(str(dataset))+'\n')
                f.write('img_name:%s'%(img_name)+'\n')
                f.write('lim:%s'%(str(lim))+'\n')
                f.write('pos:%s'%(str(pos))+'\n')

