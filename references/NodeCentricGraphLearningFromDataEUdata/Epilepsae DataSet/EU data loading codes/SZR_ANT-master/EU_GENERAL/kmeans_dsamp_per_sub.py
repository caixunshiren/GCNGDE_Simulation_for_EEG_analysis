""" This function loads the subsampled data in train_ftrs_aes.npz, extracts all the samples from a particular
   subject and then:
   -applies kmeans to reduce the sample size to n_obs/downsample_fact
   -saves the downsampled data to disk with a name like kdownsampled_1096.npz """
# Libraries
import numpy as np
# import scipy.io as sio
# import os
# import pickle
# import ieeg_funcs as ief
# import re
# import dgFuncs as dg
from sklearn.cluster import KMeans
#import matplotlib.pyplot as plt
import sys


def kmeans_downsample(ftrs,downsample_fact):
    """ ftrs = observations x ftr """
    n_obs=ftrs.shape[0]
    print('%d observations, %d ftrs' % (n_obs,ftrs.shape[1]))
    k=int(np.round(n_obs/downsample_fact))
    print('Trying %d clusters' % k)
    kclusters = KMeans(n_clusters=k).fit(ftrs)
    n_obs_per_clust=np.zeros(k)
    for a in range(k):
        n_obs_per_clust[a]=np.sum(kclusters.labels_==a)
    return kclusters, k, n_obs_per_clust


if len(sys.argv)==1:
    print('Usage: kmeans_dsamp_per_sub.py sub_id')
    exit()
if len(sys.argv)!=2:
    raise Exception('Error: kmeans_dsamp_per_sub.py requires 1 argument: sub_id')

sub=int(sys.argv[1])

# Load raw ftrs
#ftr_fname='/Users/davidgroppe/Desktop/train_ftrs_aes.npz'
ftr_fname='train_ftrs_unbnded.npz'
print("loading data from %s" % ftr_fname)
npz=np.load(ftr_fname)
print(npz.keys())


#downsample_fact=10 # used for AES
#downsample_fact=150 # used for search 1
#downsample_fact=1000 # used for search 2
downsample_fact=500 # used for search 3
#downsample_fact=1 #used for search 4, bug checking
#downsample_fact=750
#sub_list=npz['train_subs_list'][:2]
# sub_list=npz['train_subs_list']
print('Downsampling factor=%d' % downsample_fact)
# Figure out how much data to preallocate
n_downsamp_obs=0
n_dim=npz['ftrs'].shape[1]

ictal_bool=np.multiply(npz['sub_id']==sub,npz['szr_class']==1)
nonictal_bool=np.multiply(npz['sub_id']==sub,npz['szr_class']==0)
n_ictal=np.sum(ictal_bool)
n_nonictal=np.sum(nonictal_bool)
print('Sub %d, n_ictal=%d, n_nonictal=%d' % (sub,n_ictal,n_nonictal))
n_downsamp_obs+=int(np.round(n_ictal/downsample_fact))+int(np.round(n_nonictal/downsample_fact))
print('# of total downsampled windows %d' % n_downsamp_obs)

# Preallocate memory
#['ftrs', 'szr_class', 'sub_id', 'train_subs_list']
ftrs_dsamp=np.zeros((n_downsamp_obs,n_dim))
sub_ids_dsamp=np.zeros(n_downsamp_obs)
szr_class_dsamp=np.zeros(n_downsamp_obs)
dsamp_wts=np.zeros(n_downsamp_obs)

obs_ct=0
print('Working on sub %d' % sub)
ictal_bool = np.multiply(npz['sub_id'] == sub, npz['szr_class'] == 1)
if downsample_fact>1:
    # Cluster ictal observations
    print('Clustering ictal observations')
    ictal_kclusters, k, n_obs_per_clust=kmeans_downsample(npz['ftrs'][ictal_bool,:],downsample_fact)
else:
    ictal_obs=npz['ftrs'][ictal_bool,:]
    k=n_ictal
    n_obs_per_clust=np.ones(n_ictal)
obs_ct_stop=obs_ct+k
sub_ids_dsamp[obs_ct:obs_ct_stop]=sub
if downsample_fact>1:
    ftrs_dsamp[obs_ct:obs_ct_stop,:]=ictal_kclusters.cluster_centers_
else:
    ftrs_dsamp[obs_ct:obs_ct_stop, :]=ictal_obs
szr_class_dsamp[obs_ct:obs_ct_stop]=1
dsamp_wts[obs_ct:obs_ct_stop]=n_obs_per_clust/(2*np.sum(n_obs_per_clust))
obs_ct+=k

nonictal_bool = np.multiply(npz['sub_id'] == sub, npz['szr_class'] == 0)
if downsample_fact>1:
    # Cluster non-ictal observations
    print('Clustering NONictal observations')
    nonictal_kclusters, k, n_obs_per_clust=kmeans_downsample(npz['ftrs'][nonictal_bool,:],downsample_fact)
else:
    nonictal_obs=npz['ftrs'][nonictal_bool,:]
    k=n_nonictal
    n_obs_per_clust=np.ones(n_nonictal)
obs_ct_stop=obs_ct+k
sub_ids_dsamp[obs_ct:obs_ct_stop]=sub
if downsample_fact>1:
    ftrs_dsamp[obs_ct:obs_ct_stop,:]=nonictal_kclusters.cluster_centers_
else:
    ftrs_dsamp[obs_ct:obs_ct_stop, :] = nonictal_obs
szr_class_dsamp[obs_ct:obs_ct_stop]=0
dsamp_wts[obs_ct:obs_ct_stop]=n_obs_per_clust/(2*np.sum(n_obs_per_clust))
obs_ct+=k

print('Done')

# Save results to disk
out_fname='kdownsampled_'+str(sub)
print('Saving file as %s' % out_fname)
np.savez(out_fname,ftrs_dsamp=ftrs_dsamp,szr_class_dsamp=szr_class_dsamp,dsamp_wts=dsamp_wts)



# plt.figure(3)
# plt.clf()
# plt.subplot(2,1,1)
# plt.plot(sub_ids_dsamp)
# # plt.imshow(ftrs_dsamp)
#
# plt.subplot(2,1,2)
# plt.plot(szr_class_dsamp)
#
# plt.figure(4)
# plt.clf()
# plt.imshow(ftrs_dsamp.T)
#
# print('done')

