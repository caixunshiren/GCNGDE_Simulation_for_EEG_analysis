# # Sketch of patient-specific EU classifier
import numpy as np
# import sys
import pandas as pd
import os
import pickle
# import scipy.io as sio
import ieeg_funcs as ief
# import re
import dgFuncs as dg
from sklearn import preprocessing
# from scipy import stats
# from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from sklearn import svm, linear_model
from sklearn.externals import joblib

# Define sub & feature
sub='1096'
#ftr_names=['EU_MAG_LAG0','EU_MAG_LAG2']
ftr_names=['EU_MAG_LAG0','EU_MAG_LAG2','EU_MAG_LAG4','EU_MAG_LAG6']
#ftr_name='EU_MAG_LAG0'
model_type='logreg'
n_ftr_types=len(ftr_names)


# Get key directories
dir_dict=ief.get_path_dict()
print(dir_dict.keys())
ftrs_root=dir_dict['ftrs_root']
print(ftrs_root)
meta_dir=dir_dict['eu_meta']


# Get list of training and validation files
split_fname=os.path.join(meta_dir,'data_splits_FR_'+sub+'.pkl')
print('Loading %s' % split_fname)
split_dict=pickle.load(open(split_fname,'rb'))
print(split_dict.keys())
train_files=split_dict['train_files']
train_szr_files=split_dict['train_szr_files']
valid_files=split_dict['valid_files']
n_train_file=len(train_files)
n_train_szr_file=len(train_szr_files)
print('%d training files (%d contain szrs)' % (n_train_file, n_train_szr_file))
n_valid_file=len(valid_files)
print('%d validation files' % n_valid_file)


# Load first training feature file from each feature type to determine feature dimensionality
n_ftr_dim = 0
for temp_ftr_type in ftr_names:
    ftr_path=os.path.join(ftrs_root,temp_ftr_type,sub)
    ext_fname = os.path.join(ftr_path, 'ext.txt')  # Read the file extension for this type of feature
    f = open(ext_fname, 'r')
    ext = f.readline().strip()
    full_fname = os.path.join(ftr_path, train_szr_files[0] + ext)
    ftr_dict = np.load(full_fname)
    n_ftr_dim += ftr_dict['ftrs'].shape[0]
print('Ftr dimensionality is %d' % n_ftr_dim)


# Load normalization parameters
ftr_nrm_dicts=list()
for temp_ftr_type in ftr_names:
    ftr_path=os.path.join(ftrs_root,temp_ftr_type,sub)
    nrm_fname=os.path.join(ftr_path,'ftr_nrms.npz')
    print('Loading mns and sd for feature normalization from %s' % temp_ftr_type)
    # /Users/davidgroppe/PycharmProjects/SZR_ANT/FTRS/EU_MAG_LAG0/1096/ftr_nrms.npz
    ftr_nrm_dicts.append(np.load(nrm_fname))


###### IMPORT VALIDATION DATA
# Figure out how many validation time points there are to preallocate memory
n_valid_wind = 0
class_path=os.path.join(ftrs_root,'EU_SZR_CLASS',sub)
for ieeg_fname in valid_files:
    full_fname = os.path.join(class_path, ieeg_fname + '_szr_class.npz')
    class_dict = np.load(full_fname)
    n_valid_wind += len(class_dict['szr_class'])

print('%d total validation time windows' % n_valid_wind)

# Actually load the validation data
valid_ftrs = np.zeros((n_ftr_dim, n_valid_wind))  # preallocate memory
valid_class = np.ones(n_valid_wind, dtype='int8')
ftr_dim_ct=0
for ftr_type_ct, temp_ftr_type in enumerate(ftr_names):
    wind_ct = 0
    ftr_path=os.path.join(ftrs_root,temp_ftr_type,sub)
    ext_fname = os.path.join(ftr_path, 'ext.txt')  # Read the file extension for this type of feature
    f = open(ext_fname, 'r')
    ext = f.readline().strip()
    for szr_fname in valid_files:
        full_fname = os.path.join(ftr_path, szr_fname + ext)
        ftr_dict = np.load(full_fname)

        temp_ftr_dim, temp_n_ictal_wind = ftr_dict['ftrs'].shape
        if ftr_type_ct==0:
            full_fname = os.path.join(class_path, szr_fname + '_szr_class.npz')
            class_dict = np.load(full_fname)
            valid_class[wind_ct:wind_ct + temp_n_ictal_wind] = class_dict['szr_class']

        valid_ftrs[ftr_dim_ct:ftr_dim_ct+temp_ftr_dim, wind_ct:wind_ct + temp_n_ictal_wind] = ftr_dict['ftrs']
        wind_ct += temp_n_ictal_wind

    # Normalize validation data ftrs
    for ftr_loop in range(temp_ftr_dim):
        valid_ftrs[ftr_dim_ct+ftr_loop, :] = (valid_ftrs[ftr_dim_ct+ftr_loop, :] -
                               ftr_nrm_dicts[ftr_type_ct]['nrm_mn'][ftr_loop]) / ftr_nrm_dicts[ftr_type_ct]['nrm_sd'][ftr_loop]
    ftr_dim_ct+=temp_ftr_dim

print('Done loading validation data!')

plt.figure(1)
plt.clf()
plt.subplot(2,1,1)
plt.plot(valid_ftrs.T)
plt.ylabel('Z')
plt.title('Ftrs')
plt.xlim([0, n_valid_wind])

plt.subplot(2,1,2)
plt.plot(valid_class)
plt.ylabel('Szr=1, Nonszr=0')
plt.xlabel('Time Pts')
plt.xlim([0, n_valid_wind])

print('Ploting validation features as temp_valid_ftrs.jpg')
plt.savefig('temp_valid_ftrs.jpg')



# Load model
#model_list=pickle.load('temp_bagger.pkl')
model_list=joblib.load('temp_bagger.pkl')
# model_list=list()
# model_list.append(joblib.load('temp_bagger.pkl'))
n_ensemble=len(model_list)
# loaded_model = joblib.load('temp_bagger_uni.pkl')
# n_ensemble=1
print('# of models in ensemble: %d' % n_ensemble)

# print('LOADING VALID DATA FROM NPZ')
# del valid_ftrs
# bro=np.load('train_var_ftrs.npz')
# valid_ftrs=bro['valid_ftrs']
# bro=np.load('train_var_class.npz')
# valid_class=bro['valid_class']

#Apply ensemble to validation data
valid_class_hat=np.zeros(n_valid_wind)
for ensemble_ct in range(n_ensemble):
    valid_class_hat+=model_list[ensemble_ct].predict(valid_ftrs.T)
    #valid_class_hat += loaded_model.predict(valid_ftrs.T)
valid_class_hat=valid_class_hat/n_ensemble
#np.savez('load_var.npz',valid_class_hat=valid_class_hat)
#np.savez('load_var.npz',valid_ftrs=valid_ftrs)
np.savez('load_var.npz',valid_class=valid_class)
print('Ensemble validation data performance:')
jive=(valid_class_hat>=0.5)==valid_class #sign of prediction is what counts
valid_sens=np.sum(jive[valid_class==1])/np.sum(valid_class==1)
print('Sensitivity %f' % valid_sens)
valid_spec=np.sum(jive[valid_class==0])/np.sum(valid_class==0)
print('Specificity %f' % valid_spec)
valid_bal_acc=(valid_sens+valid_spec)/2
print('Balanced Accuracy=%f' % valid_bal_acc)

# print('LOADING AGAIN!!')
# del loaded_model
# loaded_model = joblib.load('temp_bagger_uni.pkl')
# n_ensemble=1
# print('# of models in ensemble: %d' % n_ensemble)
#
# #Apply ensemble to validation data
# valid_class_hat=np.zeros(n_valid_wind)
# for ensemble_ct in range(n_ensemble):
#     #valid_class_hat+=model_list[ensemble_ct].predict(valid_ftrs.T)
#     valid_class_hat += loaded_model.predict(valid_ftrs.T)
# valid_class_hat=valid_class_hat/n_ensemble
# #np.savez('load_var.npz',valid_class_hat=valid_class_hat)
# #np.savez('load_var.npz',valid_ftrs=valid_ftrs)
# np.savez('load_var.npz',valid_class=valid_class)
# print('Ensemble validation data performance:')
# jive=(valid_class_hat>=0.5)==valid_class #sign of prediction is what counts
# valid_sens=np.sum(jive[valid_class==1])/np.sum(valid_class==1)
# print('Sensitivity %f' % valid_sens)
# valid_spec=np.sum(jive[valid_class==0])/np.sum(valid_class==0)
# print('Specificity %f' % valid_spec)
# valid_bal_acc=(valid_sens+valid_spec)/2
# print('Balanced Accuracy=%f' % valid_bal_acc)