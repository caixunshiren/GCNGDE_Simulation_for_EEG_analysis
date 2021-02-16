# Computes spectral magnitude features for later classification using IIR filtering, moving window, and Hilbert transform
# Then it smoothes spectral magnitude estimates with an exponentially decaying moving average
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

# TOOD fix comments below
# Features:
# central window
# 1 second window, ends 1 sec previous
# mean of 10 second window, ends 1 sec previous
# wind_len=1 TODO use these?
# 1 second window, ends 1 sec previous
# prev_wind_width=(1, 1, 10) # two 1 second windows and a 10 second window)
# prev_wind_delay=(0, 1, 1) # centered window, and two 1 sec before center windows

# 10 second window, ends 1 sec previous
# Time to use:
# -all preictal data
# -10 seconds past onset window

if len(sys.argv)==1:
    print('Usage: compute_hilbert_mag_lag_EU.py decay_factor ')
    exit()
if len(sys.argv)!=3:
    raise Exception('Error: compute_hilbert_mag_lag_EU.py requires 2 argument: decay_factor patient_id')

decay_fact=int(sys.argv[1])
print('Extent of causal moving average is %d' % decay_fact)

sub=sys.argv[2]
print('Patient being processed is %s' % sub)

# Get paths
path_dict=ief.get_path_dict()
eu_root_dir=os.path.join(path_dict['eu_root'],'FR_'+sub)

# Create feature directory if it doesn't already exist
ftrs_root = path_dict['ftrs_root']
ftr_root = os.path.join(ftrs_root, 'EU_MAG_LAG' + str(decay_fact), sub)
if not os.path.isdir(ftr_root):
    print('Creating directory %s' % ftr_root)
    pre_root=os.path.join(ftrs_root, 'EU_MAG_LAG' + str(decay_fact));
    if not os.path.isdir(pre_root):
        os.mkdir(pre_root)
    os.mkdir(ftr_root)


# Create a file that indicates what the extension is for all features of this type
ext_file=os.path.join(ftr_root,'ext.txt')
print('Creating file %s' % ext_file)
text_file=open(ext_file,'w')
text_file.write('_bpmag_lag'+str(decay_fact)+'.npz\n')
text_file.close()


# Define frequency bands
bands=list()
bands.append((0,4)) # Delta
bands.append((4,8)) # Theta
bands.append((8,13)) # Alpha
bands.append((13,30)) # Beta
bands.append((30,50)) # Gamma
bands.append((70,100)) # High Gamma
band_labels=['DeltaMag','ThetaMag','AlphaMag','BetaMag','GammaMag','HGammaMag']
ftr_list=[]
n_band=len(bands)

# Get list of ieeg mat files
mat_fnames_dirty=os.listdir(eu_root_dir)
mat_fnames=list()
# Remove non-*.mat files
for fname in mat_fnames_dirty:
    if fname.endswith('.mat'):
        mat_fnames.append(fname)

# Loop over files
for fname in mat_fnames[:2]: # TODO make these all files

    # Import matfile
    print('Loading file %s' % os.path.join(eu_root_dir, fname))
    szr_mat = sio.loadmat(os.path.join(eu_root_dir, fname))
    Sf=szr_mat['Fs'][0][0]
    tpts_sec = szr_mat['tpts_sec'][0, :]
    n_chan=len(szr_mat['use_chans'])
    n_chan=1
    print('%d channels' % n_chan)

    # Pre-allocate memory
    # n_chan, n_tpt=szr_mat['ieeg256'].shape
    # print('%d channels, %d tpts' % (n_chan, n_tpt))

    # Extract freq band magnitude via IIR filtering and Hilbert transform
    wind_len = int(np.round(Sf))
    wind_step = int(np.round(Sf / 10))
    # TODO use all data
    abs_mag, hilb_inst_freq, hilb_sec = ief.bp_hilb_mag(szr_mat['ieeg256'][:n_chan,:10000], Sf, wind_len,
                                                         wind_step,
                                                         tpts_sec, bands)
    # abs_mag, sgram_sec=ief.bp_pwr(ieeg[onset_chan_id:onset_chan_id+1,:], Sf, wind_len, wind_step,
    #                          n_tapers, tpts_sec, bands, taper='slepian')
    # abs_mag = np.squeeze(abs_mag)
    n_hilb_tpt=len(hilb_sec)
    print('abs_mag.shape={}'.format(abs_mag.shape))
    abs_mag=np.reshape(abs_mag,(n_chan*n_band, n_hilb_tpt))
    print('abs_mag.shape={}'.format(abs_mag.shape))

    # Create feature labels TODO verify that they are correct
    if not ftr_list:
        for chan_ct in range(n_chan):
            chan=str(szr_mat['use_chans'][chan_ct,0][0])
            for band_lab in band_labels:
                ftr_list.append(chan+'_'+band_lab)
        print('ftr_list={}'.format(ftr_list))

    # Smooth data with exponential decaying moving average
    if decay_fact > 0:
        n_hilby_tpt = len(hilb_sec)
        alph = 1 / (2 ** decay_fact)
        for tpt in range(1, n_hilby_tpt):
            abs_mag[:, tpt] = (1 - alph) * abs_mag[:, tpt - 1] + alph * abs_mag[:, tpt]

    # Find szr onset(s) in the hilbert transformed data
    # szr_class=np.zeros(n_hilb_tpt,dtype='int8')
    # if np.sum(szr_mat['is_szr'][:,0])>0:
    #     for hi

    # Discard tpts affected by data edge? TODO


    # Save features
    fname_stem=fname.split('.')[0]
    ftr_fname=os.path.join(ftr_root,fname_stem+'_bpmag_lag'+str(decay_fact)+'.npz')
    print('Saving features to file %s' % ftr_fname)
    #np.savez(ftr_fname,peri_ictal=peri_ictal,time_wind_sec=sgram_sec,abs_mag=abs_mag)
    np.savez(ftr_fname, time_wind_sec=hilb_sec, ftrs=abs_mag, ftr_list=ftr_list)

    # Clear big variables
    del abs_mag, szr_mat


print('Done!')