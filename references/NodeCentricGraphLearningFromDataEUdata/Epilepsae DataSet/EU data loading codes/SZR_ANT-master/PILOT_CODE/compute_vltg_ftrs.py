# # Computes spectral power features for later classification
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
#use_subs.txt TODO use this text file
#sub_list=['NA']
#sub_list=['CO']
path_dict=ief.get_path_dict()
use_subs_df=pd.read_csv(os.path.join(path_dict['szr_ant_root'],'use_subs.txt'),header=None,na_filter=False)
ieeg_root = path_dict['ieeg_root']

for sub in use_subs_df.iloc[:,0]:
    # Get list of files to use from clinician onset csv file
    onset_csv_dir=path_dict['onset_csv']
    onset_csv_fname=os.path.join(onset_csv_dir,sub+'_clinician_onset_offset.csv')
    print('Importing file %s' % onset_csv_fname)
    onset_df=pd.read_csv(onset_csv_fname)

    use_ser=onset_df['USE4CLASSIFIER']
    use_szrs=list()
    # onset_chans=list()
    for row_id, quality in enumerate(use_ser):
        if quality=='use':
            szr_name=sub+'_d'+str(onset_df.iloc[row_id,0])+'_sz'+str(onset_df.iloc[row_id,1])
            use_szrs.append(szr_name)
    n_use_szrs=len(use_szrs)
    print('%d usable szrs:' % n_use_szrs)
    print(use_szrs)

    # Import channel names
    chan_labels=ief.import_chan_labels(sub)
    # ftr_list=list() #TODO USE THESE?
    # ftr_class_list=list()

    # Loop over usable files
    for szr_ct, szr_name in enumerate(use_szrs):
        # Load data
        in_fname=os.path.join(ieeg_root,sub,'EEG_MAT',szr_name+'.mat')
        if not os.path.isfile(in_fname):
            # Must be a subclinical seizure
            in_fname=os.path.join(ieeg_root,sub,'EEG_MAT',szr_name+'_sc.mat')
        szr_name_full=in_fname.split('/')[-1]
        ieeg, Sf, tpts_sec=ief.import_ieeg(szr_name_full)
    
        # Get time point and channel of szr onset
        onset_tpt, onset_chan=ief.clin_onset_tpt_and_chan(szr_name, onset_df)
        onset_chan_id = chan_labels.index(onset_chan)
        onset_sec=tpts_sec[onset_tpt]
    
        # Run DFT
        wind_len=Sf
        wind_step=Sf/10
        vltg_ftrs, moving_wind_sec, ftr_list=ief.cmpt_vltg_ftrs(ieeg[onset_chan_id:onset_chan_id+1,:],
                                                            wind_len, wind_step, tpts_sec)
        vltg_ftrs=np.squeeze(vltg_ftrs)
        # Trim norm
        dg.trimmed_normalize(vltg_ftrs,.4)
    
        # Record class of each time point
        n_wind=len(moving_wind_sec)
        sgram_onset_id=dg.find_nearest(moving_wind_sec,onset_sec-4)
        sgram_term_id=dg.find_nearest(moving_wind_sec,onset_sec+9)
        peri_ictal=np.ones(n_wind,dtype=np.int8)
        peri_ictal[:sgram_onset_id]=0
        peri_ictal[sgram_term_id:]=-1 # ictal (and potentially post-ictal data) that should
        # not be used for classification
    
        # output data
        ftrs_root=path_dict['ftrs_root']
        ftr_path=os.path.join(ftrs_root,'VLTG',sub)
        if not os.path.isdir(ftr_path):
            os.mkdir(ftr_path)
        ftr_fname=os.path.join(ftr_path,szr_name+'_vltg.npz')
        print('Saving features to file %s' % ftr_fname)
        np.savez(ftr_fname,peri_ictal=peri_ictal,time_wind_sec=moving_wind_sec,ftrs=vltg_ftrs,
                 ftr_list=ftr_list)


print('Done!')