# This function applies a classifier to all the clips of an epilepsiae patient & calculates:
#  -hypothetical stimulations
#  -# of seizures stimulated (both clinical and subclinical)
#  -false positive rate (i.e., stimulations outside of clinical or subclinical szrs)
#  -latency of closest stimulation to seizure onset
#
# Results are saved to the classifier's (i.e., model's) subdirectory
#
# Notes:
#  -This saves results using refractory period in seconds. The AES version of the code used minutes. Thus you can tell them apart

# Libraries
import numpy as np
import scipy.io as sio
import os
import pandas as pd
import pickle
import sys
import ieeg_funcs as ief
import dgFuncs as dg
import matplotlib.pyplot as plt
import scipy.io as sio

def replace_periods(word):
    for letter in word:
        if letter == ".":
            word = word.replace(letter,"-")
    return word

## Start of main function
if len(sys.argv)==1:
    print('Usage: compute_acc_with_refractory.py sub_id max_edm_wt model_name plot_em')
    exit()
if len(sys.argv)!=5:
    raise Exception('Error: compute_acc_with_refractory.py requires 3 arguments: sub_id max_edm_wt model_name plot_em')

# Import Parameters from command line
sub = int(sys.argv[1])
#max_edm_wt=float(sys.argv[2]) # edm_wt>=0, yhat(t+1)=yhat(t+1)*(1/2^edm_wt)+yhat(t)(1-1/2^edm_wt); bigger values of edm_wt mean more smoothing
max_edm_wt=int(sys.argv[2])
model_name=sys.argv[3]
plot_em=sys.argv[4]

for edm_wt in range(max_edm_wt+1):
    now_wt=1/np.power(2,edm_wt)
    past_wt=1-now_wt
    stim_thresh=0.5 # threshold for triggering stimulation
    Fs = 9.84615384615  # sampling rate of moving spectral energy window
    time_step=1/Fs
    # Set stim refractory period
    #refract_min=0.5 # length of refractory period in minutes
    refract_sec=30 # length of refractory period in seconds
    refract_tpt=np.round(Fs*refract_sec) #length of refractory period in time points


    # Get list of yhat files
    path_dict=ief.get_path_dict()
    if sys.platform=='linux':
        yhat_path = os.path.join('/home/dgroppe/EU_YHAT_RAW/',str(sub)+'_'+model_name)
        label_path =os.path.join('/home/dgroppe/EU_Y/',str(sub)+'_all_labels')
    else:
        yhat_path = os.path.join('/Users/davidgroppe/ONGOING/EU_YHAT_RAW/',str(sub)+'_'+model_name)
        label_path =os.path.join('/Volumes/SgateOSExJnld/EU_TEMP/',str(sub)+'_all_labels')

    # Accuracy metrics
    total_hrs = 0
    n_false_pos = 0

    # Get list of clips from data_on_off_FR*.csv
    csv_path=os.path.join(path_dict['eu_meta'],'IEEG_ON_OFF')
    csv_fname='data_on_off_FR_'+str(sub)+'.csv'
    on_off_df=pd.read_csv(os.path.join(csv_path,csv_fname))

    edge_pts=1178 # number of initial time points skipped due to EDM edge effects

    # Loop over header files
    #temp_files=['109600102_0000_yhat.npz','109600102_0001_yhat.npz','109600102_0002_yhat.npz','109600102_0072_yhat.npz']
    #for f in temp_files:
    clin_szr_ct=0
    # first_hdr_file=True
    stim_sec=list()
    #DataFrame.sort_values(by, axis=0, ascending=True, inplace=False
    on_off_df.sort_values(by=['StartSec'], axis=0, ascending=True, inplace=True)
    #on_off_df=on_off_df.sort_values(by=['StartSec'],axis=0,ascending=True,inplace=False)
    last_stim = on_off_df['StartSec'].iloc[0]-refract_sec-1 # initialize so that stimulation can begin at start of first clip
    n_clips=on_off_df.shape[0]
    # print(last_stim)
    # print(on_off_df['StartSec'][0])
    # print(on_off_df['StartSec'].iloc[0])
    # print(np.min(on_off_df['StartSec']))
    # print(np.max(on_off_df['StartSec']))
    # print(on_off_df.head())
    # plt.figure(1)
    # plt.clf()
    # plt.plot(on_off_df['StartSec'])
    # plt.show()
    # exit()
    #for hdr_ct, hdr_fname in enumerate(on_off_df['HeaderFname']):
    print("now_wt {}, past_wt {}".format(now_wt,past_wt)) # ??
    for hdr_ct in range(n_clips):
        # Note that the files should already be sorted in chronological order in on_off_df
        hdr_fname=on_off_df['HeaderFname'].iloc[hdr_ct]
        root_fname=hdr_fname.split('.')[0]

        # Load classifier output for this clip
        # yhat_fname=root_fname+'_yhat.npz'
        #print('Analyzing file %s' % yhat_fname)
        # yhat_npz = np.load(os.path.join(yhat_path, yhat_fname))
        yhat_fname=root_fname+'_yhat.mat'
        print('Analyzing file %s' % yhat_fname)
        yhat_npz = sio.loadmat(os.path.join(yhat_path, yhat_fname))
        if np.isnan(yhat_npz['max_yhat'][0,0])==False:
            # File is long enough to have EDM features and classifier outputs
            #??yhat_edm=np.zeros(yhat_npz['yhat_soz_chans'].shape)
            #n_wind=yhat_npz['max_yhat'].shape[1]
            n_chan, n_wind=yhat_npz['yhat_soz_chans'].shape
            yhat_edm=np.nan_to_num(yhat_npz['yhat_soz_chans'])
            yhat_max=np.zeros(n_wind)
            for tloop in range(1,n_wind):
                for cloop in range(n_chan):
                    yhat_edm[cloop,tloop]=yhat_edm[cloop,tloop]*now_wt+past_wt*yhat_edm[cloop,tloop-1]
                yhat_max[tloop]=np.max(yhat_edm[:,tloop])

            # Compute # of seconds in file
            file_dur_hr = n_wind / (Fs * 3600)
            total_hrs += file_dur_hr
            #print('total_hrs=%f' % total_hrs)

            # Load clinician labels for this clip
            y_fname = str(sub) + '_y_' + root_fname + '.mat'
            label_f = os.path.join(label_path, y_fname)
            #print('Loading file %s' % y_fname)
            label_mat = sio.loadmat(label_f)

            time_ptr=on_off_df['StartSec'].iloc[hdr_ct]+edge_pts/Fs # First feature time point
            # if first_hdr_file==True:
            #     time_ptr=0 #set to start of first file
            #     first_hdr_file==False

            # Compute hypothetical stimulations with refractory periods
            stim = np.zeros(n_wind)
            #  for wind_ct, yhat in enumerate(yhat_npz['max_yhat'][0,:]):
            for wind_ct, yhat in enumerate(yhat_max):
                if wind_ct>0:
                    time_ptr+=time_step
                if yhat > stim_thresh and ((time_ptr-last_stim) > refract_sec):
                    # Hypotheticaly stimulate
                    stim[wind_ct] = 1
                    stim_sec.append(time_ptr)
                    last_stim = time_ptr


            # Compute false positives
            se_szr_class = np.squeeze(label_mat['se_szr_class'])
            # print('Min se_szr_class: %f' % np.min(se_szr_class))
            # Note, this ignores subclinical szrs (se_szr_class=-1) and clinical szrs (se_szr_class=1)
            nonszr_bool = se_szr_class == 0 # TODO extend this 5 sec before clinician onset,
            # should probably make a separate matlab variable
            #print('false+ %d' % np.sum(stim[nonszr_bool]))
            n_false_pos += np.sum(stim[nonszr_bool])

        if False:
            # Plot szr class and hypothetical stimulations for this clip
            plt.figure(1)
            plt.clf()
            #plt.plot(yhat_npz['max_yhat'],'b-')
            plt.plot(yhat_max, 'b-')
            #plt.plot(stim[stim>0],'r.')
            plt.plot(stim,'r.')
            plt.plot(se_szr_class,'g-')
            xlim=plt.xlim()
            plt.plot(xlim,[0.5, 0.5],'k:')
            plt.show()


    ### PROCESS SENSITIVITY & LATENCY OF STIM TO SEIZURES
    # Load list of szrs
    szr_times_fname = os.path.join(path_dict['szr_ant_root'],'EU_METADATA','SZR_TIMES/',
                                   'szr_on_off_FR_' + str(sub) + '.pkl')
    szr_df = pickle.load(open(szr_times_fname, 'rb'))
    n_szr = szr_df.shape[0]

    # np.savez('sbox.npz',stim_sec=stim_sec)
    # Loop over szrs and:
    stim_lat = np.zeros(n_szr)
    clin_szr = np.zeros(n_szr)
    szr_hit = np.zeros(n_szr)
    stim_sec_np = np.asarray(stim_sec)
    for szr_id in range(n_szr):
        onset_sec = szr_df['SzrOnsetSec'][szr_id]
        offset_sec = szr_df['SzrOffsetSec'][szr_id]
        #TODO deal with szrs witout offsets
        if szr_df['SzrType'][szr_id] == 'Clinical':
            clin_szr[szr_id] = 1

        # print('Szr %d, %s, Onset %f  Offset %f sec' % (szr_id,szr_df['SzrType'][szr_id],onset_sec,offset_sec))
        # exit()

        # 1) find the stimulation that is closest in time to seizure onset
        nearest_id = dg.find_nearest(stim_sec, onset_sec)
        # nearest_id=dg.find_nearest(stim_sec_np,onset_sec)
        stim_lat[szr_id] = stim_sec[nearest_id] - onset_sec
        #         temp_id = np.argmin(np.abs(np.asarray(stim_ids) - onset_id))
        #         closest_id = stim_ids[temp_id]

        # 2) see if stimulation happens SOMEwhere in the target window
        # stim_bool=(stim_lat>=onset_sec) and (stim_lat<=offset_sec)
        stim_bool = np.multiply(stim_sec_np >= onset_sec - 5, stim_sec_np <= offset_sec)
        if np.sum(stim_bool) > 0:
            szr_hit[szr_id] = 1

        print('%s Szr %d, lat %f, hit %d, duration=%f' % (szr_df['SzrType'][szr_id],szr_id, stim_lat[szr_id], szr_hit[szr_id], offset_sec - onset_sec))

    n_clin_szr=int(np.sum(clin_szr))
    print('%d clinical szrs' % n_clin_szr)
    print('%d subclinical szrs' % np.sum(clin_szr == 0))
    clin_bool=clin_szr==1
    print('%d/%d clinical szrs stimulated' % (np.sum(szr_hit[clin_bool]), np.sum(clin_bool)))
    print('%d/%d subclinical szrs stimulated' % (np.sum(szr_hit[clin_bool==False]), np.sum(clin_bool==False)))

    fp_per_hour = n_false_pos / total_hrs
    print('%f of false positives/hr' % fp_per_hour)
    print('%f of false positives/day' % (fp_per_hour*24))

    # Save results to disk
    outpath=os.path.join(path_dict['szr_ant_root'],'MODELS',model_name)
    if False:
        # Save results as npz file
        outfname=str(sub)+'_edmwt_'+replace_periods(str(edm_wt))+'_refract_'+replace_periods(str(refract_sec))+'_stim_results'
        print('Saving results to %s' % os.path.join(outpath,outfname))
        np.savez(os.path.join(outpath,outfname),
             stim_lat=stim_lat,
             stim_sec=stim_sec,
             clin_szr=clin_szr,
             szr_hit=szr_hit,
             total_hrs=total_hrs,
             n_false_pos=n_false_pos,
             fp_per_hour=fp_per_hour,
             stim_thresh=stim_thresh,
             edm_wt=edm_wt,
             refract_sec=refract_sec)
    else:
        # Save results as mat file
        outfname=str(sub)+'_edmwt_'+replace_periods(str(edm_wt))+'_refract_'+replace_periods(str(refract_sec))+'_stim_results.mat'
        print('Saving results to %s' % os.path.join(outpath,outfname))
        out_dict={"stim_lat": stim_lat,
             "stim_sec": stim_sec,
             "clin_szr": clin_szr,
             "szr_hit": szr_hit,
             "total_hrs": total_hrs,
             "n_false_pos": n_false_pos,
             "fp_per_hour": fp_per_hour,
             "stim_thresh": stim_thresh,
             "edm_wt": edm_wt,
             "refract_sec": refract_sec}
        sio.savemat(os.path.join(outpath,outfname),out_dict)


    if plot_em in ['True', 'true']:
        plt.figure(1)
        plt.clf()
        plt.boxplot(stim_lat[clin_bool])
        plt.ylabel('Seconds')
        plt.xticks([])
        plt.title('Stim Latency relative to Clin Szr Onset')
        xlim=[.85, 1.15]
        plt.xlim(xlim)
        plt.plot(xlim,[0, 0],'k:')
        plt.xlim(xlim)
        mn_lat=np.mean(stim_lat[clin_bool])
        plt.plot([.95, 1.05],[mn_lat, mn_lat],'r-',linewidth=3)
        sd_lat=np.std(stim_lat[clin_bool])
        plt.plot([1, 1],[mn_lat-sd_lat, mn_lat+sd_lat],color='purple',linewidth=3)
        plt.plot(np.ones(n_clin_szr)+(np.random.rand(n_clin_szr)-0.5)/25,stim_lat[clin_bool],'b.')
        plt.show()