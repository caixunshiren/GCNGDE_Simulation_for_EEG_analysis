# Plots ALL the seizures (both clinical and subclinical) from a single patient in random order via a strat plot
# Data are converted to avg ref before plotting
# MATLAB has already pruned bad channels and downsampled the data to 256 Hz

import numpy as np
import numpy.matlib
import pandas as pd
import os
import scipy.io as sio
import ieeg_funcs as ief
import re
import dgFuncs as dg
import sys

from sklearn import preprocessing
import sys
from scipy import stats
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt


if len(sys.argv)==1:
    print('Usage: strat_plot_of_szrs.py patient# (e.g., python strat_plot_of_szrs.py 1096)')
    exit()
if len(sys.argv)!=2:
    raise Exception('Error: strat_plot_of_szrs.py requires 1 argument')

# Define sub and count number of szr files
# Directory where data are
path_dict=ief.get_path_dict()
eu_root_dir=path_dict['eu_root']

sub='1096'
sub_dir=os.path.join(eu_root_dir,'FR_'+sub)
print('Reading mat fnames from %s' % sub_dir)
mat_fnames=list()
for fname in os.listdir(sub_dir):
    if fname.endswith('.mat'):
        mat_fnames.append(fname)

n_szrs=len(mat_fnames)
print('%d szr files for sub %s' % (n_szrs,sub))


# TODO: use all mat_fnames
#for szr_ct, szr_fname in enumerate(mat_fnames):
#for szr_ct, szr_fname in enumerate(mat_fnames[:2]):
perm_ids=np.random.permutation(n_szrs)
for szr_ct in perm_ids:
    szr_fname=mat_fnames[szr_ct]
    print('Visualizing szr %d/%d' % (szr_ct+1,n_szrs))
    # Load Szr
    szr_mat=sio.loadmat(os.path.join(sub_dir,szr_fname))
    szr_type=szr_mat['szr_type'][0]
    print('File %s, szr type is %s' % (szr_fname,szr_type))

    # Find onset & Offset
    Fs=szr_mat['Fs'][0][0]
    tpts_sec = szr_mat['tpts_sec'][0, :]
    n_tpt = len(tpts_sec)
    szr_ids=np.argwhere(szr_mat['is_szr'][:,0]>0)
    onset_id=int(szr_ids[0])
    offset_id=int(szr_ids[-1])
    print('%f sec of preonset data' % ((onset_id-1)/Fs))

    n_chan=len(szr_mat['use_chans'][0,:])
    print('%d channels' % n_chan)
    chan_labels=list()
    for chan_ct in range(n_chan):
        chan_labels.append(szr_mat['use_chans'][0,chan_ct][0])

    # convert data to avg ref
    ieeg256=szr_mat['ieeg256']
    mn_ieeg256=np.mean(ieeg256,axis=0)
    ieeg256=ieeg256-np.matlib.repmat(mn_ieeg256, n_chan, 1)

    plt.figure(1)
    plt.clf()
    # strat_plot(dat,chan_labels,fig_id=1,show_chan_ids=None,h_offset=2,srate=1,tpts_sec=None,fontsize=9):
    # ief.strat_plot(ieeg256[:,(onset_id-Fs*10):(offset_id+Fs*10)],chan_labels,fig_id=1,
    #                show_chan_ids=np.arange(int(n_chan/10)),h_offset=700,srate=Fs);
    use_ids=np.arange((onset_id-Fs*10),(offset_id+Fs*10))
    ief.strat_plot(ieeg256[:,use_ids],chan_labels,fig_id=1,
               show_chan_ids=np.arange(n_chan),h_offset=700,srate=Fs,tpts_sec=tpts_sec[use_ids])
    ylim=plt.ylim()
    plt.plot([tpts_sec[onset_id], tpts_sec[onset_id]],ylim,'k--')
    plt.plot([tpts_sec[offset_id], tpts_sec[offset_id]],ylim,'k:')
    plt.ylim(ylim)
    plt.title('Szr '+str(szr_ct)+' '+szr_type+' '+szr_fname)
    plt.show()
