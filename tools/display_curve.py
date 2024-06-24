


import glob
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

plt.switch_backend('agg')

if __name__=='__main__':
    prf_path = "plots"
    filename = "CrackLS315.pdf"
    output  = os.path.join("plots", filename)
    eval_mode = "prf"
    files = glob.glob(os.path.join(prf_path, "*.{}".format(eval_mode)))
    _, axs = plt.subplots(nrows=1, ncols=1, figsize=(5,5))

    for ff in files:
        fname = ff.split('/')[-1].split('.')[0]
        p_acc, r_acc, f_acc = [], [], []
        with open(ff, 'r') as fin:
            for ll in fin:
                bt, p, r, f = ll.strip().split('\t')
                p_acc.append(float(p))
                r_acc.append(float(r))
                f_acc.append(float(f))
        max_index = np.argmax(np.array(f_acc))
        axs.plot(np.array(r_acc), np.array(p_acc), label='[F={:.03f}]{}'.format(f_acc[max_index], fname).replace('=0.', '=.'), lw=2)

    axs.grid(True, linestyle='-.')
    axs.set_xlim([0, 1.])
    axs.set_ylim([0, 1.])
    axs.set_xlabel('{}'.format("recall"))
    axs.set_ylabel('{}'.format("precision"))
    axs.legend(loc='{}'.format("lower left"))

    pdf = PdfPages(r'{}'.format(output))
    plt.savefig(pdf, format='pdf', bbox_inches='tight')
    pdf.close()
    pdf=None
