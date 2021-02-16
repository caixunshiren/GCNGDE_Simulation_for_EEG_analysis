# Computes seizure class for moving window hilbert transformed data
# Note that seizure onsets are EXTENDED 4 SECONDS INTO THE FUTURE so that the classifier can anticipate onset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
import os
import ieeg_funcs as ief
import dgFuncs as dg
import pickle
import sys
from scipy import signal

if len(sys.argv)==1:
    print('Usage: compute_szr_anticipate_class_EU.py patient_id')
    exit()
if len(sys.argv)!=2:
    raise Exception('Error: compute_szr_anticipate_class_EU.py requires 1 argument: patient_id')

sub=sys.argv[1]
print('Patient being processed is %s' % sub)

# Get paths
path_dict=ief.get_path_dict()
eu_root_dir=os.path.join(path_dict['eu_root'],'FR_'+sub)

# Create feature directory if it doesn't already exist
ftrs_root = path_dict['ftrs_root']
ftr_root = os.path.join(ftrs_root, 'EU_SZR_ANT_CLASS', sub)
if not os.path.isdir(ftr_root):
    print('Creating directory %s' % ftr_root)
    pre_root=os.path.join(ftrs_root, 'EU_SZR_ANT_CLASS')
    if not os.path.isdir(pre_root):
        os.mkdir(pre_root)
    os.mkdir(ftr_root)


# Get list of ieeg mat files
mat_fnames_dirty=os.listdir(eu_root_dir)
mat_fnames=list()
# Remove non-*.mat files
for fname in mat_fnames_dirty:
    if fname.endswith('.mat'):
        mat_fnames.append(fname)

# Loop over files
#for fname in mat_fnames[:1]: # use a subset of files for debugging
for fname in mat_fnames:
    # Import matfile
    print('Loading file %s' % os.path.join(eu_root_dir, fname))
    szr_mat = sio.loadmat(os.path.join(eu_root_dir, fname))
    Sf=szr_mat['Fs'][0][0]
    tpts_sec = szr_mat['tpts_sec'][0, :]
    is_szr=szr_mat['is_szr']
    orig_is_szr=np.copy(is_szr)
    n_tpt=len(tpts_sec)

    # Find the start and stop of each szr in the file
    df_szr = np.diff(is_szr.T)
    onset_ids = np.where(df_szr.T == 1)[0] + 1
    offset_ids = np.where(df_szr.T == -1)[0]
    is_szr=is_szr*-1 # Make all szr tpts -1 by default

    # Loop over szr onsets and define target as 4 sec before an 9 sec after onset
    # All other in szr classes will be ignored
    pre_onset_horizon=Sf*4
    post_onset_horizon = Sf * 9
    for tpt in onset_ids:
        is_szr[tpt-pre_onset_horizon:tpt+post_onset_horizon]=1

    # Plot to double check
    # if sum(orig_is_szr)>0:
    #     plt.figure(1)
    #     plt.clf()
    #     plt.plot(is_szr,'r')
    #     plt.plot(orig_is_szr,'-b')
    #     plt.show()

    wind_len = int(np.round(Sf))
    wind_step = int(np.round(Sf / 10))

    n_half_wind = int(np.round(wind_len / 2))
    n_hilby_tpt = len(np.arange(n_half_wind, n_tpt - n_half_wind, wind_step))
    szr_class = np.zeros(n_hilby_tpt,dtype='int8')
    hilb_sec = np.zeros(n_hilby_tpt)
    # Moving window
    hilb_ct = 0
    for tpt_ct in range(n_half_wind, n_tpt - n_half_wind, wind_step):
        mn_class = np.mean(is_szr[(tpt_ct - n_half_wind):(tpt_ct + n_half_wind)])
        szr_class[hilb_ct] = (mn_class>=0.5) # Note, if training a classifier with continuous outputs, there would be no need to threshold
        hilb_sec[hilb_ct] = np.mean(tpts_sec[(tpt_ct - n_half_wind):(tpt_ct + n_half_wind)])
        hilb_ct += 1

    # Save szr class
    fname_stem=fname.split('.')[0]
    class_fname=os.path.join(ftr_root,fname_stem+'_szr_class.npz')
    print('Saving seizure class of each moving window to file %s' % class_fname)
    #np.savez(class_fname,peri_ictal=peri_ictal,time_wind_sec=sgram_sec,abs_mag=abs_mag)
    np.savez(class_fname, szr_class=szr_class, time_wind_sec=hilb_sec, wind_len=wind_len, wind_step=wind_step)

    # Clear big variables
    del szr_class, szr_mat


print('Done!')