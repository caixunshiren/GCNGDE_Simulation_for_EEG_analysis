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
n_ftr_types=len(ftr_names)
model_type='logreg'

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
for ftr_type_ct, temp_ftr_type in enumerate(ftr_names):
    wind_ct=0
    ftr_path=os.path.join(ftrs_root,temp_ftr_type,sub)
    ext_fname = os.path.join(ftr_path, 'ext.txt')
    # Read the file extension for this type of feature 
    f = open(ext_fname, 'r')
    ext = f.readline().strip()
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
    ext_fname = os.path.join(ftr_path, 'ext.txt')
    # Read the file extension for this type of feature 
    f = open(ext_fname, 'r')
    ext = f.readline().strip()
    # Load data for this feature
    for ftr_fname in train_files:
        full_fname=os.path.join(ftr_path,ftr_fname+ext)
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

# Plot Training Data
plt.figure(1)
plt.clf()
plt.subplot(2,1,1)
plt.plot(train_ftrs.T)
plt.ylabel('Z')
plt.title('Ftrs')
plt.xlim([0, n_szr_wind*2])

plt.subplot(2,1,2)
plt.plot(train_class)
plt.ylabel('Szr=1, Nonszr=0')
plt.xlabel('Time Pts')
plt.xlim([0, n_szr_wind*2])

print('Ploting validation features as temp_train_ftrs.jpg')
plt.savefig('temp_train_ftrs.jpg')


##### TRAIN CLASSIFIER
print('Training classifier...')
# Train classifier
if model_type=='logreg':
    model = linear_model.LogisticRegression(C=1)
else:
    model = svm.SVC(C=1, gamma=1e-5)
model.fit(train_ftrs.T, train_class)

# Accuracy on training data
train_class_hat = model.predict(train_ftrs.T) # outputs 0 or 1
train_bal_acc, train_sens, train_spec=ief.perf_msrs(train_class, train_class_hat)
# print('Unique train_class_hat {}'.format(np.unique(train_class_hat)))
# jive=(train_class_hat == train_class)
# train_bal_acc[ensemble_ct]=np.mean(jive)
print('Training data results:')
print('Accuracy: %f' % train_bal_acc)
print('Sensitivity: %f' % train_sens)
print('Specificity: %f' % train_spec)


# Save Model
model_file = 'temp_classifier.pkl'
print('Saving model as %s' % model_file)
_ = joblib.dump(model, model_file, compress=3)



#### LOAD VALIDATION DATA & APPLY CLASSIFIER (ONE FILE AT A TIME)

# Figure out how many validation time points there are to preallocate memory
n_valid_wind = 0
class_path=os.path.join(ftrs_root,'EU_SZR_CLASS',sub)
for ieeg_fname in valid_files:
    full_fname = os.path.join(class_path, ieeg_fname + '_szr_class.npz')
    class_dict = np.load(full_fname)
    n_valid_wind += len(class_dict['szr_class'])
print('%d total validation time windows' % n_valid_wind)

# Actually load the validation data
print('Loading validation data...')
valid_ftrs = np.zeros((n_ftr_dim, n_valid_wind))  # preallocate memory
valid_class = np.ones(n_valid_wind, dtype='int8')
ftr_dim_ct=0
for ftr_type_ct, temp_ftr_type in enumerate(ftr_names):
    wind_ct = 0
    ftr_path=os.path.join(ftrs_root,temp_ftr_type,sub)
    ext_fname = os.path.join(ftr_path, 'ext.txt')  # Read the file extension for this type of feature
    f = open(ext_fname, 'r')
    ext = f.readline().strip()
    for ftr_fname in valid_files:
        full_fname = os.path.join(ftr_path, ftr_fname + ext)
        ftr_dict = np.load(full_fname)

        temp_ftr_dim, temp_n_ictal_wind = ftr_dict['ftrs'].shape
        if ftr_type_ct==0:
            full_fname = os.path.join(class_path, ftr_fname + '_szr_class.npz')
            class_dict = np.load(full_fname)
            valid_class[wind_ct:wind_ct + temp_n_ictal_wind] = class_dict['szr_class']

        valid_ftrs[ftr_dim_ct:ftr_dim_ct+temp_ftr_dim, wind_ct:wind_ct + temp_n_ictal_wind] = ftr_dict['ftrs']
        wind_ct += temp_n_ictal_wind

    # Normalize validation data ftrs
    for ftr_loop in range(temp_ftr_dim):
        valid_ftrs[ftr_dim_ct+ftr_loop, :] = (valid_ftrs[ftr_dim_ct+ftr_loop, :] -
                               ftr_nrm_dicts[ftr_type_ct]['nrm_mn'][ftr_loop]) / ftr_nrm_dicts[ftr_type_ct]['nrm_sd'][ftr_loop]
    ftr_dim_ct+=temp_ftr_dim

print('Applying classifier to validation data...')
valid_class_hat = model.predict(valid_ftrs.T) # outputs 0 or 1
valid_bal_acc, valid_sens, valid_spec=ief.perf_msrs(valid_class, valid_class_hat)
# print('Unique train_class_hat {}'.format(np.unique(train_class_hat)))
# jive=(train_class_hat == train_class)
# train_bal_acc[ensemble_ct]=np.mean(jive)
print('Validation data results:')
print('Accuracy: %f' % valid_bal_acc)
print('Sensitivity: %f' % valid_sens)
print('Specificity: %f' % valid_spec)


exit()




#print('Grand mean train acc %f' % np.mean(train_bal_acc))
# print('Grand mean validation acc %f' % np.mean(valid_bal_acc))
# print('Train bal acc={}'.format(train_bal_acc))
# print('Valid bal acc={}'.format(valid_bal_acc))

# Save model
#model_file = os.path.join(model_path, 'model' + str(left_out_id) + '.pkl')
model_file = 'temp_bagger.pkl'
print('Saving model as %s' % model_file)
_ = joblib.dump(model_list, model_file, compress=3)
#_ = joblib.dump(model_list[0], model_file, compress=3)
# joblib.dump(model, 'temp_bagger_uni.pkl')

# wts_file = 'temp_booster.npz'
# print('Saving model wts as %s' % wts_file)
# np.savez(wts_file,ensemble_wts=ensemble_wts)

#Apply ensemble to training data
# train_class_hat=np.zeros(n_szr_wind*2)
# for ensemble_ct in range(n_ensemble):
#     train_class_hat+=ensemble_wts[ensemble_ct]*(model_list[ensemble_ct].predict(train_ftrs.T)*2-1) #predictions are -1 or 1
# print('Ensemble training data performance:')
# jive=(train_class_hat>=0)==train_class #sign of prediction is what counts
# train_sens=np.sum(jive[train_class==1])/np.sum(train_class==1)
# print('Sensitivity %f' % train_sens)
# train_spec=np.sum(jive[train_class==0])/np.sum(train_class==0)
# print('Specificity %f' % train_spec)
# train_bal_acc=(train_sens+train_spec)/2
# print('Balanced Accuracy=%f' % train_bal_acc)

#Apply ensemble to validation data
valid_class_hat=np.zeros(n_valid_wind)
for ensemble_ct in range(n_ensemble):
    valid_class_hat+=model_list[ensemble_ct].predict(valid_ftrs.T)
valid_class_hat=valid_class_hat/n_ensemble
#np.savez('train_var.npz',valid_class_hat=valid_class_hat)
# np.savez('train_var.npz',valid_ftrs=valid_ftrs)
# np.savez('train_var.npz',valid_class=valid_class)
print('Ensemble validation data performance:')
jive=(valid_class_hat>=0.5)==valid_class #sign of prediction is what counts
valid_sens=np.sum(jive[valid_class==1])/np.sum(valid_class==1)
print('Sensitivity %f' % valid_sens)
valid_spec=np.sum(jive[valid_class==0])/np.sum(valid_class==0)
print('Specificity %f' % valid_spec)
valid_bal_acc=(valid_sens+valid_spec)/2
print('Balanced Accuracy=%f' % valid_bal_acc)

