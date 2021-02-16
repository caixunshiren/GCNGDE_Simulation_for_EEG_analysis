""" This script loads the hypothetical stimulations and summarizes accuracy and amount of data across
a list of patients. Plots are made too if plot_em==True"""

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os
import sys
import euGenFuncs as eu

## Start of main function
if len(sys.argv)==1:
    print('Usage: plot_stim_results.py model_name sub_list plot_em')
    exit()
if len(sys.argv)!=4:
    raise Exception('Error: plot_stim_results.py requires 3 arguments: model_name sub_list plot_em')

# Import Parameters from command line
model_name=sys.argv[1]
#print(model_name)
sub_list_fname=sys.argv[2]
print('Reading subjects to process from %s' % sub_list_fname)
plot_em=sys.argv[3]

# Load test subjects
text_file=open(sub_list_fname,'r')
temp=text_file.readlines()
subs=list()
for raw_sub in temp:
    subs.append(raw_sub.strip())


# TODO make all these command line
#plot_em=False
#subs=[264, 273, 862, 1125]
#pth='lregSeAes8_nokdsamp_1'
#pth='/home/dgroppe/GIT/SZR_ANT/MODELS/svmAesFinale_1'
if sys.platform=='linux':
    pth=os.path.join('/home/dgroppe/GIT/SZR_ANT/MODELS/',model_name)
else:
    pth = os.path.join('/Users/davidgroppe/PycharmProjects/SZR_ANT/MODELS/', model_name)
n_subs=len(subs)
try_thresh=np.arange(0.2,0.7,0.1)
n_thresh=len(try_thresh)

#preallocate mem
sens=np.zeros((n_subs, n_thresh))
total_hrs=np.zeros(n_subs)
total_clin_szrs=np.zeros(n_subs)
fp_per_day=np.zeros((n_subs, n_thresh))
mn_stim_lat=np.zeros((n_subs, n_thresh))

# Loop over files
for thresh_ct, thresh in enumerate(try_thresh):
    for sub_ct, sub in enumerate(subs):
        in_fname=str(sub)+'_thresh_0-'+str(int(10*thresh))+'_refract_30_stim_results.mat'
        npz=sio.loadmat(os.path.join(pth,in_fname))
        #in_fname=str(sub)+'_thresh_0-'+str(int(10*thresh))+'_refract_0-5_stim_results.npz'
        #npz=np.load(os.path.join(pth,in_fname))
        #sens[sub_ct,thresh_ct]=npz['sens']
        clin_szr_bool=npz['clin_szr']==1
        sens[sub_ct,thresh_ct]=np.mean(npz['szr_hit'][clin_szr_bool])
        fp_per_day[sub_ct,thresh_ct]=npz['fp_per_hour']*24
        mn_stim_lat[sub_ct,thresh_ct]=np.mean(npz['stim_lat'][clin_szr_bool])
        if thresh_ct==0:
            total_hrs[sub_ct]=npz['total_hrs']
            total_clin_szrs[sub_ct]=np.sum(clin_szr_bool)
            #total_clin_szrs[sub_ct]=npz['n_clin_szr']
#print(npz.keys())
#print('Mean SD sensitivity: %f (%f)' % (np.mean(sens),np.std(sens)))


# Make plots if requested
if plot_em in ['true', 'True']:
    # Plot Sensitivity
    plt.figure(1)
    plt.clf()
    for sub_ct in range(n_subs):
        plt.plot(try_thresh+(np.random.rand(1)-.5)/50,sens[sub_ct,:],'.',label='Sub'+str(subs[sub_ct]))
    plt.plot(try_thresh,np.mean(sens,axis=0),'ro-')
    plt.xlabel('Threshold')
    plt.ylabel('Pptn of Szrs Stimulated')
    plt.legend()

    thresh_id=3
    print('For Thresh: %f' % try_thresh[thresh_id])
    print('Mean (SD) Sensitivity: %f (%f)' % (np.mean(sens[:,thresh_id]),np.std(sens[:,thresh_id])))
    plt.show()


    # Plot False Positive Stimulations/Day
    plt.figure(2)
    plt.clf()
    plt.plot(try_thresh,np.mean(fp_per_day,axis=0),'ro-')
    for sub_ct in range(n_subs):
        plt.plot(try_thresh+(np.random.rand(1)-.5)/50,fp_per_day[sub_ct,:],'.',label='Sub'+str(subs[sub_ct]))
    plt.xlabel('Threshold')
    plt.ylabel('False postitive stim/day')
    plt.legend()
    plt.show()
    print('For Thresh: %f' % try_thresh[thresh_id])
    print('Mean (SD) False Positive Stim/Day: %f (%f)' % (np.mean(fp_per_day[:,thresh_id]),np.std(fp_per_day[:,thresh_id])))


    # Plot Mean Stim Latency for each Clinical Szr
    plt.figure(3)
    plt.clf()
    plt.plot(try_thresh,np.mean(mn_stim_lat,axis=0),'ro-')
    for sub_ct in range(n_subs):
        plt.plot(try_thresh+(np.random.rand(1)-.5)/50,mn_stim_lat[sub_ct,:],'.',label='Sub'+str(subs[sub_ct]))
    plt.xlabel('Threshold')
    plt.ylabel('Seconds to Onset')
    plt.legend()
    plt.show()
    print('For Thresh: %f' % try_thresh[thresh_id])
    print('Mean (SD) Stim Latency: %f (%f) seconds' % (np.mean(mn_stim_lat[:,thresh_id]),
                                                          np.std(mn_stim_lat[:,thresh_id])))


# Output key results as text
# Report total hours of EEG & # of clinical szrs
print('Total hours of test data: %f' % np.sum(total_hrs))
print('Total days of test data: %f' % np.sum(total_hrs/24))
print('Mean (SD) days of test data: %f (%f)' % (np.mean(total_hrs/24),np.std(total_hrs/24)))
print('Total # of clinical szrs: %d' % np.sum(total_clin_szrs))
print('Mean (SD) # of clinical szrs: %f (%f)' % (np.mean(total_clin_szrs),np.std(total_clin_szrs)))

print()
thresh_id=3 # Stim threshold of 0.5
print('For Thresh: %f' % try_thresh[thresh_id])
print('Mean (SD) Stim Latency: %f (%f) seconds' % (np.mean(mn_stim_lat[:,thresh_id]),
                                                      np.std(mn_stim_lat[:,thresh_id])))
print('Mean (SD) False Positive Stim/Day: %f (%f)' % (np.mean(fp_per_day[:,thresh_id]),np.std(fp_per_day[:,thresh_id])))
print('Mean (SD) Sensitivity: %f (%f)' % (np.mean(sens[:,thresh_id]),np.std(sens[:,thresh_id])))

