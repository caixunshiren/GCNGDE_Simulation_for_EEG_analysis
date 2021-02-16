import numpy as np
import pandas as pd
import os
import scipy.io
import ieeg_funcs as ief
import re
import dgFuncs as dg
from sklearn import preprocessing
import sys
from scipy import stats
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

if len(sys.argv)<2:
    raise ValueError('You need to pass sub codename as first argument')

sub=sys.argv[1]

# Define sub and onset channel
# TODO add this path_dict
#onset_df=pd.read_csv('/Users/davidgroppe/PycharmProjects/SZR_ANT/chans_of_interest.csv',na_filter=False)
#row_id=onset_df[onset_df.iloc[:,0]==sub].index.tolist()
#onset_chan=onset_df.iloc[row_id[0],1].strip() #strip removes white space
#print('Sub=%s, onset chan=%s' % (sub,onset_chan))
print('Sub=%s' % sub)

# Get list of mat files
path_dict = ief.get_path_dict()
ieeg_root = path_dict['ieeg_root']
mat_file_path=os.path.join(ieeg_root,sub,'EEG_MAT')
mat_file_list=list()
for f in os.listdir(mat_file_path):
    if f.endswith('.mat') and f.startswith(sub+'_d'):
        print(f)
        mat_file_list.append(f)
n_files=len(mat_file_list)
print('n_files=%d' % n_files)

# Load manual onset times
# csv_path='/Users/davidgroppe/ONGOING/SZR_SPREAD/PATIENTS/'+sub+'/ONSETTIMES/'
# print('Importing manual onset times from %s' % csv_path)
# csv_list=list()
# for f in os.listdir(csv_path):
#     if f.endswith('manualOnsets.txt'):
#         csv_list.append(f)
#
# n_files=len(csv_list)
# print('%d manual wide-band onset files found' % n_files)

# Import Clinician Szr Onset Times
# TODO add this path_dict
# if sys.platform=='linux':
#     onset_csv_dir='/home/dgroppe/TWH_INFO/CLINICIAN_ONSET_TIMES'
# else:
#     onset_csv_dir='/Users/davidgroppe/Dropbox/TWH_INFO/CLINICIAN_ONSET_TIMES'
onset_csv_dir=path_dict['onset_csv']
onset_csv_fname=os.path.join(onset_csv_dir,sub+'_clinician_onset_offset.csv')
print('Importing file %s' % onset_csv_fname)
onset_df=pd.read_csv(onset_csv_fname)

onset_df.head()

# Get list of channels and ID of onset channel
chan_labels=ief.import_chan_labels(sub)
# onset_chan_id=chan_labels.index(onset_chan)

n_chan=len(chan_labels)

figure_path = os.path.join(path_dict['pics'],'ONSET_ACTIVITY',sub)
#figure_path=os.path.join('/Users/davidgroppe/PycharmProjects/SZR_ANT/PICS/ONSET_ACTIVITY',sub)
if not os.path.exists(figure_path):
    os.makedirs(figure_path)

# Loop through manual files (since I have one for each mat file)
#for man_file_loop in range(1,2): #??
for man_file_loop in range(n_files):
    #szr_name=csv_list[man_file_loop].split('_manualOnsets')[0]
    szr_name=mat_file_list[man_file_loop].split('.')[0]

    # See if I have a clinician onset
    print(szr_name)
    onset_tpt, onset_chan=ief.clin_onset_tpt_and_chan(szr_name, onset_df)

    if onset_chan=='?':
        print('Warning: %s does not have a DG specified onset channel.' % szr_name)
        print('Ignoring this szr for the time being')
        onset_tpt=np.nan
    elif onset_tpt<0:
        print('Warning: %s has a clinician onset time that is earlier than the file start time.' % szr_name)
        print('Ignoring this szr for the time being')
        onset_tpt=np.nan
    elif np.isnan(onset_tpt):
        print('Warning: %s has no clinician onset time.'  % szr_name)
        print('Ignoring this szr for the time being')
    else:
        # Get onset channel for this szr
        onset_chan_id = chan_labels.index(onset_chan)

        # Load the ieeg data
        ieeg, Sf, tpts_sec=ief.import_ieeg(szr_name+'.mat')
        
        # Calcuate onset window (there is uncertainty due second resolution of Xltek text files)
        onset_upper_bnd_sec=(onset_tpt/Sf)+1
        onset_lower_bnd_sec=(onset_tpt/Sf)-1
    
        # Plot voltage time series at all channels and onset
        [h, ax]=ief.strat_plot(ieeg,chan_labels,tpts_sec=tpts_sec)
        ylim=ax.get_ylim()
        plt.plot([onset_upper_bnd_sec, onset_upper_bnd_sec],ylim,'k--')
        plt.plot([onset_lower_bnd_sec, onset_lower_bnd_sec],ylim,'k--')
        plt.title(szr_name)
        plt.ylabel('Voltage')
        plt.savefig(os.path.join(figure_path, szr_name + '_voltage_strat.pdf'))

        # Plot voltage time series at just the onset channel
        plt.figure(2)
        plt.clf()
        plt.plot(tpts_sec,ieeg[onset_chan_id,:])
        plt.xlim([onset_lower_bnd_sec-5, onset_upper_bnd_sec+3])
        ax=plt.gca()
        ylim=ax.get_ylim()
        plt.plot([onset_upper_bnd_sec, onset_upper_bnd_sec],ylim,'r--')
        plt.plot([onset_lower_bnd_sec, onset_lower_bnd_sec],ylim,'r--')
        plt.xlabel('Seconds')
        plt.ylabel('Voltage')
        plt.title(szr_name+' '+chan_labels[onset_chan_id])
        plt.savefig(os.path.join(figure_path, szr_name + '_voltage_strat_zoom.pdf'))
    
        # Compute spectrogram at onset channel
        wind_len=Sf
        wind_step=Sf/10
        n_tapers=4
        sgram, f, sgram_sec=ief.mt_sgram(ieeg[onset_chan_id,:],Sf,wind_len,wind_step,n_tapers,tpts_sec)
        cutoff_freq=Sf*.4 # remove frequencies above anti-aliasing filter cutoff
        f=f[f<=cutoff_freq]
        n_freq=len(f)
        sgram=sgram[:n_freq,:]
        n_wind=len(sgram_sec)
    
        # Plot sgram at just onset channel with onset overlay
        plt.figure(3)
        plt.clf()
        ax = plt.gca()
        # 40% Trimmed normalization
        dg.trimmed_normalize(sgram,.4)
        #abs_mx = np.max(np.abs(sgram))
        abs_mx = np.percentile(np.abs(sgram), 99)  # use a saturated color map
        im=ax.imshow(sgram,vmin=-abs_mx,vmax=abs_mx)
        onset_sgram_tpt_lower=dg.find_nearest(sgram_sec,onset_lower_bnd_sec)
        onset_sgram_tpt_upper=dg.find_nearest(sgram_sec,onset_upper_bnd_sec)
        ylim=plt.ylim()
        plt.plot([onset_sgram_tpt_upper, onset_sgram_tpt_upper],ylim,'k--')
        plt.plot([onset_sgram_tpt_lower, onset_sgram_tpt_lower],ylim,'k--')
        raw_xticks=plt.xticks()
        xtick_labels=list()
        for tick in raw_xticks[0]:
            if tick<n_wind:
                xtick_labels.append(str(int(sgram_sec[int(tick)])))
            else:
                xtick_labels.append('noData')
        _=plt.xticks(raw_xticks[0],xtick_labels) #works
        plt.ylim(ylim)
        plt.xlim([0,n_wind])
        plt.ylabel('Hz')
        plt.xlabel('Seconds')
        plt.gca().invert_yaxis()
        plt.title(szr_name+' '+chan_labels[onset_chan_id]+' '+'Sgram (40% trim norm dB)')
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar_max_tick=int(np.floor(abs_mx))
        cbar_min_tick=-cbar_max_tick
        cbar = plt.colorbar(im, cax=cax, ticks=[cbar_min_tick, 0, cbar_max_tick])
        plt.savefig(os.path.join(figure_path, szr_name + '_onset_sgram.pdf'))
    
        # Plot sgram at just onset channel with onset overlay zoomed in on onset
        plt.figure(4)
        plt.clf()
        ax = plt.gca()
        zoom_sgram_tpt_lower=dg.find_nearest(sgram_sec,onset_lower_bnd_sec-5)
        zoom_sgram_tpt_upper=dg.find_nearest(sgram_sec,onset_upper_bnd_sec+5)
        #im=ax.imshow(sgram[:,zoom_sgram_tpt_lower:zoom_sgram_tpt_upper])
        im=ax.imshow(sgram,vmin=-abs_mx,vmax=abs_mx)
        ylim=plt.ylim()
        plt.plot([onset_sgram_tpt_upper, onset_sgram_tpt_upper],ylim,'k--')
        plt.plot([onset_sgram_tpt_lower, onset_sgram_tpt_lower],ylim,'k--')
        plt.ylim(ylim)
        plt.xlim([zoom_sgram_tpt_lower,zoom_sgram_tpt_upper])
        xtick_labels=[str(int(sgram_sec[zoom_sgram_tpt_lower])), str(int(sgram_sec[zoom_sgram_tpt_upper]))]
        plt.xticks([zoom_sgram_tpt_lower,zoom_sgram_tpt_upper],xtick_labels)
        plt.ylabel('Hz')
        plt.xlabel('Seconds')
        plt.gca().invert_yaxis()
        plt.title(szr_name+' '+chan_labels[onset_chan_id])
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar=plt.colorbar(im, cax=cax,ticks=[cbar_min_tick, 0, cbar_max_tick])
        cbar.set_label('40% trim z-score log10(pwr)')
        plt.savefig(os.path.join(figure_path, szr_name + '_onset_sgram_zoom.pdf'))

        # Compute sgram across all channels
        omni_sgram=np.zeros((n_freq*n_chan,n_wind))
        omni_sgram_yticks=np.zeros(n_chan)
        for chan_loop in range(n_chan):
            temp_sgram, temp_f, _=ief.mt_sgram(ieeg[chan_loop,:],Sf,wind_len,wind_step,n_tapers,tpts_sec)
            omni_sgram[chan_loop*n_freq:(chan_loop+1)*n_freq,:]=temp_sgram[:n_freq,:]
            omni_sgram_yticks[chan_loop]=(chan_loop*n_freq+(chan_loop+1)*n_freq)/2
        
        # Plot sgram at all channels
        omni_sgram_z=omni_sgram.copy()
        # get rid of possible inf values
        omni_sgram_z[omni_sgram_z==np.inf]=0
        omni_sgram_z[omni_sgram_z == -np.inf] = 0
        #40% Trimmed normalization
        dg.trimmed_normalize(omni_sgram_z,.4)
        #abs_mx=np.max(np.abs(omni_sgram_z))
        abs_mx=np.percentile(np.abs(omni_sgram_z),98) #use a saturated color map
        plt.figure(5)
        plt.clf()
        ax = plt.gca()
        im=ax.imshow(omni_sgram_z,vmin=-abs_mx,vmax=abs_mx,aspect='auto')
        ylim=plt.ylim()
        plt.plot([onset_sgram_tpt_upper, onset_sgram_tpt_upper],ylim,'k:')
        plt.plot([onset_sgram_tpt_lower, onset_sgram_tpt_lower],ylim,'k:')
        plt.plot([onset_sgram_tpt_upper, onset_sgram_tpt_upper],
                 [onset_chan_id*n_freq, (1+onset_chan_id)*n_freq],'r-')
        plt.plot([onset_sgram_tpt_lower, onset_sgram_tpt_lower],
                 [onset_chan_id*n_freq, (1+onset_chan_id)*n_freq],'r-')
        raw_xticks=plt.xticks()
        xtick_labels=list()
        for tick in raw_xticks[0]:
            if tick<n_wind:
                xtick_labels.append(str(int(sgram_sec[int(tick)])))
            else:
                xtick_labels.append('noData')
        _=plt.xticks(raw_xticks[0],xtick_labels) #works
        plt.ylim(ylim)
        if n_chan>36:
            plt.yticks(omni_sgram_yticks, chan_labels, fontsize=4)
        else:
            plt.yticks(omni_sgram_yticks,chan_labels,fontsize=8)
        #plt.xlim([0,n_wind])
        plt.xlim([onset_sgram_tpt_lower-250, np.min([onset_sgram_tpt_lower+250, n_wind])])
        plt.ylabel('Hz')
        plt.xlabel('Seconds')
        plt.gca().invert_yaxis()
        plt.title(szr_name+' '+'Sgram (trim norm dB)')
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar_max_tick=int(np.floor(abs_mx))
        cbar_min_tick=-cbar_max_tick
        # cbar_min_tick=int(np.floor(np.min(omni_sgram_z)*.95))
        # cbar_max_tick=int(np.floor(np.max(omni_sgram_z)*.95))
        cbar=plt.colorbar(im, cax=cax,ticks=[cbar_min_tick, 0, cbar_max_tick])
        cbar.set_label('40% trim z-score log10(pwr)')
        plt.savefig(os.path.join(figure_path, szr_name + '_omni_sgram.pdf'))
