# Subsamples non-ictal data
import numpy as np
# import sys
import pandas as pd
import os
import pickle
import ieeg_funcs as ief
import dgFuncs as dg
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import svm, linear_model
from sklearn.externals import joblib

# Define sub & feature
sub='1096'
szr_ant=True # If true, only data from 4 sec before to 9 sec are considered targets.
if szr_ant:
    print('In Seizure Anticipation Mode!!!!')
    print('Data from 4 seconds before to 9 seconds after onset are targets.')

# All other szr time points are ignored
#ftr_names=['EU_MAG_LAG0','EU_MAG_LAG2']
ftr_names=['EU_MAG_LAG0','EU_MAG_LAG2','EU_MAG_LAG4','EU_MAG_LAG6','EU_MAG_LAG8']
n_ftr_types=len(ftr_names)
print('# of ftrs: %d' % n_ftr_types)
print(ftr_names)
edge_pts=1177 # # of time pts at the start of each file to ignore due to edge effects

# Get key directories
dir_dict=ief.get_path_dict()
ftrs_root=dir_dict['ftrs_root']
meta_dir=dir_dict['eu_meta']

# Get list of training files
split_fname=os.path.join(meta_dir,'data_splits_FR_'+sub+'.pkl')
print('Loading %s' % split_fname)
split_dict=pickle.load(open(split_fname,'rb'))
train_files=split_dict['train_files']
train_szr_files=split_dict['train_szr_files']
n_train_file=len(train_files)
n_train_szr_file=len(train_szr_files)
print('%d training files (%d contain szrs)' % (n_train_file, n_train_szr_file))


# Figure out how many training data ictal time points there are to preallocate memory
if szr_ant:
    class_path = os.path.join(ftrs_root, 'EU_SZR_ANT_CLASS', sub)
else:
    class_path=os.path.join(ftrs_root,'EU_SZR_CLASS',sub)
n_szr_wind=0
for szr_fname in train_szr_files:
    full_fname=os.path.join(class_path,szr_fname+'_szr_class.npz')
    class_dict=np.load(full_fname)
    n_szr_wind+=np.sum(class_dict['szr_class'][edge_pts:])
print('%d total training data szr time windows' % n_szr_wind)

# Count # of non-ictal windows to figure out how many non-ictal samples to ideally draw from each file
n_nonszr_wind=0
for szr_fname in train_files:
    full_fname=os.path.join(class_path,szr_fname+'_szr_class.npz')
    class_dict=np.load(full_fname)
    n_nonszr_wind+=np.sum(class_dict['szr_class'][edge_pts:]==0)
print('%d total NON-szr time windows' % n_nonszr_wind)

szr_ids_dict=dict()
nonszr_ids_dict=dict()


###### IMPORT TRAINING DATA

# Import all ictal training data
# train_ftrs=np.zeros((n_ftr_dim,n_szr_wind*2)) #preallocate memory
# train_class=np.ones(n_szr_wind*2,dtype='int8')
# train_class[n_szr_wind:]=0
# ftr_dim_ct=0

# Load ictal data
pptn_per_file = n_szr_wind / n_nonszr_wind
print('Loading %f fraction of NON-ictal windows from each file' % pptn_per_file)
for szr_fname in train_szr_files:
    # Need to load szr classes in order to select just ictal time points
    full_fname = os.path.join(class_path, szr_fname + '_szr_class.npz')
    class_dict = np.load(full_fname)
    szr_ids_dict[szr_fname]=np.where(class_dict['szr_class'][edge_pts:] == 1)[0]+edge_pts

# Load NON-ictal data
wind_ct=0
for fname_ct, ftr_fname in enumerate(train_files):
    full_fname = os.path.join(class_path, ftr_fname + '_szr_class.npz')
    class_dict = np.load(full_fname)
    nonszr_ids = np.where(class_dict['szr_class'][edge_pts:] == 0)[0]+edge_pts
    #print('Max Min nonszr_ids: %d  %d' % (max(nonszr_ids), min(nonszr_ids)))

    temp_n_nonictal_wind = len(nonszr_ids)
    n_load_wind = int(np.round(temp_n_nonictal_wind * pptn_per_file))
    if fname_ct == (n_train_file - 1):
        # If this is the last file, load exactly enough windows to equal the #
        # of szr windows
        n_load_wind = n_szr_wind-wind_ct
    rand_wind_ids = nonszr_ids[np.random.randint(0, temp_n_nonictal_wind, n_load_wind)]
    nonszr_ids_dict[ftr_fname]=rand_wind_ids
    wind_ct+=n_load_wind
# print(len(szr_ids_dict[szr_fname]))
# print(len(nonszr_ids_dict[ftr_fname]))

if szr_ant:
    pickle.dump(szr_ids_dict, open('szr_ant_ids_dict.pkl', 'wb'))
    pickle.dump(nonszr_ids_dict, open('nonszr_ant_ids_dict.pkl', 'wb'))
    pickle.dump(n_szr_wind, open('n_szr_ant_wind.pkl', 'wb'))
else:
    pickle.dump(szr_ids_dict,open('szr_ids_dict.pkl','wb'))
    pickle.dump(nonszr_ids_dict,open('nonszr_ids_dict.pkl','wb'))
    pickle.dump(n_szr_wind,open('n_szr_wind.pkl','wb'))
print('Done selecting training IDs')