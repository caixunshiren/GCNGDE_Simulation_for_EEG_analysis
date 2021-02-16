# # Computes stdev of spectral power features for later classification
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
import os
import ieeg_funcs as ief
import dgFuncs as dg
import pickle


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

# Load list of subs to use
lag=3 # extent of moving average window in units of seconds
print('Extent of causal moving window is %d seconds!' % lag)
path_dict=ief.get_path_dict()
use_subs_df=pd.read_csv(os.path.join(path_dict['szr_ant_root'],'use_subs.txt'),header=None,na_filter=False)
ieeg_root = path_dict['ieeg_root']

for sub in use_subs_df.iloc[:,0]:
    # Define input and output directories
    ftr_root = '/Users/davidgroppe/PycharmProjects/SZR_ANT/FTRS'
    one_sec_path = os.path.join(path_dict['ftrs_root'], 'PWR', sub)
    lag_root_path=os.path.join(path_dict['ftrs_root'], 'PWRSTD_' + str(lag) + 'SEC')
    if not os.path.isdir(lag_root_path):
        print('Creating directory: %s' % lag_root_path)
        os.mkdir(lag_root_path)
    lag_path = os.path.join(lag_root_path, sub)
    if not os.path.isdir(lag_path):
        print('Creating directory: %s' % lag_path)
        os.mkdir(lag_path)
    else:
        print('Output directory: %s' % lag_path)

    # Loop over 1 second window files
    for f in os.listdir(one_sec_path):
        print('Loading file %s' % f)
        one_sec_dat = np.load(os.path.join(one_sec_path, f))
        time_wind_sec = one_sec_dat['time_wind_sec']
        peri_ictal = one_sec_dat['peri_ictal']

        #Add suffix to feature names
        one_sec_ftr_list = one_sec_dat['ftr_list']
        ftr_list=[]
        for ftr_label in one_sec_ftr_list:
            ftr_list.append(ftr_label + 'Std3Sec')
            #ftr_list[ftr_ct]=ftr_list[ftr_ct]+'3SEC'

        # Preallocate mm
        n_dim, n_wind = one_sec_dat['ftrs'].shape
        causal_std_pwr = np.zeros((n_dim, n_wind))

        # Compute sampling rate
        dlt = time_wind_sec[1] - time_wind_sec[0]
        wind_hz = 1 / dlt
        n_meta_wind = int(lag * wind_hz)

        # Apply causal moving average
        for wind in range(n_wind):
            use_ids = np.arange(wind - n_meta_wind, wind + 1, dtype=int)  # avg between lag back and current time point
            use_ids = use_ids[use_ids >= 0]  # remove any window before beginning of file
            # use_ids=use_ids.astype(int)
            causal_std_pwr[:, wind] = np.log(1+np.std(one_sec_dat['ftrs'][:, use_ids], axis=1)) # log trans to make data
            # more normally distributed

        dg.trimmed_normalize(causal_std_pwr, .4)

        # Save file
        f_stem = f.split('.')[0]
        out_fname = os.path.join(lag_path, f_stem + '_std' + str(lag) + 'sec.npz')
        print('Saving file %s' % out_fname)
        np.savez(out_fname,
                 time_wind_sec=time_wind_sec,
                 ftrs=causal_std_pwr,
                 ftr_list=ftr_list,
                 peri_ictal=peri_ictal)



print('Done!')