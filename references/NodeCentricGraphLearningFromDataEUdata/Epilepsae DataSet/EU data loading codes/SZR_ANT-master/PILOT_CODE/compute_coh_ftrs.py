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

# Define frequency bands
bands=list()
bands.append((1,4)) # Delta
bands.append((4,8)) # Theta
bands.append((8,13)) # Alpha
bands.append((13,30)) # Beta
bands.append((30,50)) # Gamma
bands.append((70,100)) # High Gamma
ftr_list=['DeltaCoh','ThetaCoh','AlphaCoh','BetaCoh','GammaCoh','HGammaCoh']

for sub in use_subs_df.iloc[:,0]: #TODO revert to this to use all subs
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

        # Adjust upper frequency band based on sampling rate
        bands[5]=(70,Sf*.4)
    
        # Get time point and channel of szr onset
        onset_tpt, onset_chan=ief.clin_onset_tpt_and_chan(szr_name, onset_df)
        onset_chan_id = chan_labels.index(onset_chan)
        onset_sec=tpts_sec[onset_tpt]
    
        # Run DFT
        wind_len=Sf
        wind_step=Sf/10
        bp_coh, sgram_sec=ief.bp_coh_omni(ieeg, onset_chan_id, Sf, wind_len, wind_step,
                                 tpts_sec, bands)
        # bp_coh, sgram_sec=ief.bp_pwr(ieeg[onset_chan_id:onset_chan_id+1,:], Sf, wind_len, wind_step,
        #                          n_tapers, tpts_sec, bands, taper='slepian')
        bp_coh=np.squeeze(bp_coh)
        # Trim norm
        dg.trimmed_normalize(bp_coh,.4)
    
        # Record class of each time point
        n_wind=len(sgram_sec)
        sgram_onset_id=dg.find_nearest(sgram_sec,onset_sec-4)
        sgram_term_id=dg.find_nearest(sgram_sec,onset_sec+9)
        peri_ictal=np.ones(n_wind,dtype=np.int8)
        peri_ictal[:sgram_onset_id]=0
        peri_ictal[sgram_term_id:]=-1 # ictal (and potentially post-ictal data) that should
        # not be used for classification
    
        # output data
        ftrs_root=path_dict['ftrs_root']
        ftr_path=os.path.join(ftrs_root,'COH',sub)
        if not os.path.isdir(ftr_path):
            os.mkdir(ftr_path)
        ftr_fname=os.path.join(ftr_path,szr_name+'_bpcoh.npz')
        print('Saving features to file %s' % ftr_fname)
        #np.savez(ftr_fname,peri_ictal=peri_ictal,time_wind_sec=sgram_sec,bp_coh=bp_coh)
        np.savez(ftr_fname, peri_ictal=peri_ictal, time_wind_sec=sgram_sec, ftrs=bp_coh, ftr_list=ftr_list)


print('Done!')