# # Computes spectral magnitude features for later classification using IIR filtering, moving window, and Hilbert transform
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
import os
import ieeg_funcs as ief
import dgFuncs as dg
import pickle
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


# Load list of subs to use
#sub_list=['NA']
#sub_list=['CO']
path_dict=ief.get_path_dict()
#use_subs_df=pd.read_csv(os.path.join(path_dict['szr_ant_root'],'use_subs.txt'),header=None,na_filter=False)
use_subs_df=pd.read_csv('use_subs.txt',header=None,na_filter=False)
ieeg_root = path_dict['ieeg_root']

# Define frequency bands
bands=list()
bands.append((0,4)) # Delta
bands.append((4,8)) # Theta
bands.append((8,13)) # Alpha
bands.append((13,30)) # Beta
bands.append((30,50)) # Gamma
bands.append((70,100)) # High Gamma
ftr_list=['DeltaMag','ThetaMag','AlphaMag','BetaMag','GammaMag','HGammaMag']

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

        # Downsample data to 250 Hz if need be
        orig_tpts_sec=tpts_sec.copy()
        if Sf == 500:
            print('Downsampling data to 250 Hz')
            ieeg = signal.decimate(ieeg, 2, axis=1, zero_phase=True)
            tpts_sec=signal.decimate(tpts_sec, 2, axis=0, zero_phase=True)
            Sf = 250
        elif Sf != 250:
            raise ValueError('Sf needs to be 500 or 250')

        # Get time point and channel of szr onset
        onset_orig_tpt, onset_chan=ief.clin_onset_tpt_and_chan(szr_name, onset_df)
        onset_chan_id = chan_labels.index(onset_chan)

        # Need to recompute time point of clinician onset due to possible change in sampling rate
        onset_sec=orig_tpts_sec[onset_orig_tpt]
        onset_tpt=dg.find_nearest(tpts_sec,onset_sec)

        # Detrend data just in case. Some patients have a crazy trend in the first channel
        ieeg = signal.detrend(ieeg)

        # Extract freq band magnitude via IIR filtering and Hilbert transform
        wind_len=int(np.round(Sf))
        wind_step=int(np.round(Sf/10))
        abs_mag, hilb_inst_freq, sgram_sec=ief.bp_hilb_mag(ieeg[onset_chan_id:onset_chan_id+1,:], Sf, wind_len, wind_step,
                                          tpts_sec, bands)
        # abs_mag, sgram_sec=ief.bp_pwr(ieeg[onset_chan_id:onset_chan_id+1,:], Sf, wind_len, wind_step,
        #                          n_tapers, tpts_sec, bands, taper='slepian')
        abs_mag=np.squeeze(abs_mag)
        # Trim norm
        dg.trimmed_normalize(abs_mag,.4)
    
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
        ftr_path=os.path.join(ftrs_root,'MAG',sub)
        if not os.path.isdir(ftr_path):
            os.mkdir(ftr_path)
        ftr_fname=os.path.join(ftr_path,szr_name+'_bpmag.npz')
        print('Saving features to file %s' % ftr_fname)
        #np.savez(ftr_fname,peri_ictal=peri_ictal,time_wind_sec=sgram_sec,abs_mag=abs_mag)
        np.savez(ftr_fname, peri_ictal=peri_ictal, time_wind_sec=sgram_sec, ftrs=abs_mag, ftr_list=ftr_list,
                 onset_chan=onset_chan,onset_chan_id=onset_chan_id)


print('Done!')