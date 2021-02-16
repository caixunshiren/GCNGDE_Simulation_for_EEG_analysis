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
#ftr_names=['EU_MAG_LAG2']
#ftr_names=['EU_MAG_LAG0','EU_MAG_LAG2']
ftr_names=['EU_MAG_LAG0','EU_MAG_LAG2','EU_MAG_LAG4','EU_MAG_LAG6','EU_MAG_LAG8']
#ftr_name='EU_MAG_LAG0'
n_ftr_types=len(ftr_names)
model_type='logreg'
C=1
# model_type='svm'
# C=351.119173
# gam=0.000040
szr_ant=True
edge_pts=1177 # # of time pts at the start of each file to ignore due to edge effects

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
test_files=split_dict['test_files']
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

# Load indices of training points
n_szr_wind=pickle.load(open('n_szr_wind.pkl','rb'))
n_train_wind=n_szr_wind*2
print('# of training data windows: %d' % n_train_wind)
szr_ids_dict=pickle.load(open('szr_ids_dict.pkl','rb'))
nonszr_ids_dict=pickle.load(open('nonszr_ids_dict.pkl','rb'))


#### IMPORT TRAINING DATA
train_ftrs=np.zeros((n_ftr_dim,n_train_wind)) #preallocate memory
train_class=np.ones(n_train_wind,dtype='int8')
train_class[n_szr_wind:]=0 # Last half of data are training

# Load Szr Data
ftr_dim_ct=0
ext_list=[]
for ftr_type_ct, temp_ftr_type in enumerate(ftr_names):
    wind_ct=0
    ftr_path=os.path.join(ftrs_root,temp_ftr_type,sub)
    ext_fname = os.path.join(ftr_path, 'ext.txt')
    # Read the file extension for this type of feature 
    f = open(ext_fname, 'r')
    ext = f.readline().strip()
    ext_list.append(ext)
    # Load data for this feature
    for ftr_fname in train_szr_files:
        full_fname=os.path.join(ftr_path,ftr_fname+ext)
        ftr_dict=np.load(full_fname)
        temp_ftr_dim= ftr_dict['ftrs'].shape[0]

        temp_szr_ids=szr_ids_dict[ftr_fname]
        temp_n_ictal_wind=len(temp_szr_ids)
        train_ftrs[ftr_dim_ct:ftr_dim_ct+temp_ftr_dim,wind_ct:wind_ct+temp_n_ictal_wind]=ftr_dict['ftrs'][:,temp_szr_ids]
        wind_ct+=temp_n_ictal_wind

    # Note feature normalization is done after non-szr data are loaded
    ftr_dim_ct += temp_ftr_dim

# Load NON-szr data
ftr_dim_ct=0
ftr_dim_list=list()
for ftr_type_ct, temp_ftr_type in enumerate(ftr_names):
    wind_ct=n_szr_wind
    ftr_path=os.path.join(ftrs_root,temp_ftr_type,sub)
    # Load data for this feature
    for ftr_fname in train_files:
        #full_fname=os.path.join(ftr_path,ftr_fname+ext)
        full_fname = os.path.join(ftr_path, ftr_fname + ext_list[ftr_type_ct])
        ftr_dict=np.load(full_fname)
        temp_ftr_dim= ftr_dict['ftrs'].shape[0]

        temp_nonszr_ids=nonszr_ids_dict[ftr_fname]
        temp_n_nonictal_wind=len(temp_nonszr_ids)
        train_ftrs[ftr_dim_ct:ftr_dim_ct+temp_ftr_dim,wind_ct:wind_ct+temp_n_nonictal_wind]=ftr_dict['ftrs'][:,temp_nonszr_ids]
        wind_ct+=temp_n_nonictal_wind
    ftr_dim_ct += temp_ftr_dim
    ftr_dim_list.append(temp_ftr_dim)

print('Done loading training data!')

# Normalize training features
ftr_dim_ct=0
for ftr_type_ct, temp_ftr_type in enumerate(ftr_names):
    # Normalize validation data ftrs
    for ftr_loop in range(ftr_dim_list[ftr_type_ct]):
        train_ftrs[ftr_dim_ct+ftr_loop, :] = (train_ftrs[ftr_dim_ct+ftr_loop, :] -
                               ftr_nrm_dicts[ftr_type_ct]['nrm_mn'][ftr_loop]) / ftr_nrm_dicts[ftr_type_ct]['nrm_sd'][ftr_loop]
    ftr_dim_ct +=ftr_dim_list[ftr_type_ct]

print('Done normalizing training data!')

# #Plot Training Data
# plt.figure(1)
# plt.clf()
# plt.subplot(2,1,1)
# plt.plot(train_ftrs.T)
# plt.ylabel('Z')
# plt.title('Ftrs')
# plt.xlim([0, n_szr_wind*2])
#
# plt.subplot(2,1,2)
# plt.plot(train_class)
# plt.ylabel('Szr=1, Nonszr=0')
# plt.xlabel('Time Pts')
# plt.xlim([0, n_szr_wind*2])
#
# print('Ploting validation features as temp_train_ftrs.jpg')
# plt.savefig('temp_train_ftrs.jpg')


#### TRAIN CLASSIFIER
# Train classifier
if model_type=='logreg':
    model = linear_model.LogisticRegression(C=C)
else:
    model = svm.SVC(C=C, gamma=gam)
model.fit(train_ftrs.T, train_class)

# Accuracy on training data
train_class_hat = model.predict(train_ftrs.T) # outputs 0 or 1
train_bal_acc, train_sens, train_spec=ief.perf_msrs(train_class, train_class_hat)
print('Subsampled training data results:')
print('Accuracy: %f' % train_bal_acc)
print('Sensitivity: %f' % train_sens)
print('Specificity: %f' % train_spec)

# Save Model
model_file = 'temp_classifier.pkl'
print('Saving model as %s' % model_file)
_ = joblib.dump(model, model_file, compress=3)

valid_bal_acc, valid_sens, valid_spec, valid_acc=ief.apply_model_2_file_list(model, valid_files, ftr_names,
                                                                             ftr_nrm_dicts, sub, n_ftr_dim,
                                                                             ext_list,edge_pts,szr_ant)

print('Validation data results:')
print('Raw accuracy: %f' % valid_acc)
print('Balanced accuracy: %f' % valid_bal_acc)
print('Sensitivity: %f' % valid_sens)
print('Specificity: %f' % valid_spec)


# Apply model to all test data
test_bal_acc, test_sens, test_spec, test_acc=ief.apply_model_2_file_list(model, test_files, ftr_names,
                                                                             ftr_nrm_dicts, sub, n_ftr_dim,
                                                                         ext_list, edge_pts,szr_ant)
print('Test data results:')
print('Raw accuracy: %f' % test_acc)
print('Balanced accuracy: %f' % test_bal_acc)
print('Sensitivity: %f' % test_sens)
print('Specificity: %f' % test_spec)


# Apply model to all training data
tfull_bal_acc, tfull_sens, tfull_spec, tfull_acc=ief.apply_model_2_file_list(model, train_files, ftr_names,
                                                                             ftr_nrm_dicts, sub, n_ftr_dim,
                                                                             ext_list,edge_pts,szr_ant)

print('FULL training data results:')
print('Raw accuracy: %f' % tfull_acc)
print('Balanced accuracy: %f' % tfull_bal_acc)
print('Sensitivity: %f' % tfull_sens)
print('Specificity: %f' % tfull_spec)



