# Loop over all the ~1 hr blocks of data and plot them as a strat plot to see if there are bad time periods
import numpy as np
import numpy.matlib
import pandas as pd
import os
import scipy.io as sio
import ieeg_funcs as ief
import dgFuncs as dg

from sklearn import preprocessing
import sys
from scipy import stats
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt


if len(sys.argv)==1:
    print('Usage: plot_all_data_strat_plot.py patient_id')
    exit()
if len(sys.argv)!=2:
    raise Exception('Error: plot_all_data_strat_plot.py requires 1 argument: patient_id')

sub=sys.argv[1]
print('Patient being processed is %s' % sub)

mat_fnames=list()
in_path=os.path.join('/Users/davidgroppe/ONGOING/EU_EEG/','FR_'+sub)
for fname in os.listdir(in_path):
    if fname.endswith('.mat'):
        mat_fnames.append(fname)

n_mat_files=len(mat_fnames)
print('# of mat files %d' % len(mat_fnames))

rand_file_ids=np.random.permutation(n_mat_files)
#for a in range(n_mat_files):
for a in range(1):
#for a in rand_file_ids[:10]: # plot a random subset
    print('Loading file %s' % mat_fnames[a])
    mat=sio.loadmat(os.path.join(in_path,mat_fnames[a]))
    
    chan_labels=list()
    for b in range(len(mat['use_chans'])):
        chan_labels.append(mat['use_chans'][b,:][0][0])
#     print('chan_labels {}'.format(chan_labels))
    
    plt.figure(1)
    plt.clf()
    ief.strat_plot(mat['ieeg256'],chan_labels,fig_id=1,show_chan_ids=None,h_offset=3000,
           srate=mat['Fs'],tpts_sec=mat['tpts_sec'].T,fontsize=9);
    plt.title(mat_fnames[a]);
    out_fname=os.path.join('PNG',mat_fnames[a].split('.')[0]+'.png')
    print('Saving figure to %s' % out_fname)
    plt.savefig(out_fname)




