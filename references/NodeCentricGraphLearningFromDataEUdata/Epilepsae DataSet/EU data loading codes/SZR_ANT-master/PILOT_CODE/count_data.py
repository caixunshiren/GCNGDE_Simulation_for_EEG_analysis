# This script simply counts the amount of data and outputs summary statistics to the command line. For example:
# Total # of dimensions: 40
# Total # of time windows: 137142
# Total # of files: 61
# Mean [SD] # of szrs: 7.625000 [4.635124]
# [ 168.899       193.898       122.56466667  154.998       123.01664706
#   105.299       175.23133333  144.149     ]
# Mean [SD] # sec of preictal dta: 148.506956 [28.375134]

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
import os
import ieeg_funcs as ief
import dgFuncs as dg
from sklearn import linear_model
from sklearn.externals import joblib
import sys
import json
from shutil import copyfile

if len(sys.argv)==1:
    print('Usage: train_ensemble.py params.json')
    exit()
if len(sys.argv)!=2:
    raise Exception('Error: train_ensemble.py requires 1 argument: params.json')

# Import parameters from json file
param_fname=sys.argv[1]
print('Importing model parameters from %s' % param_fname)
with open(param_fname) as param_file:
    params=json.load(param_file)

try_C=[]
# try_C.append(float(np.sys.argv[1]))
try_C.append(float(params['C']))
print('C value set to %f' % try_C[0])
# C = SVM regularization parameter, the smaller it is, the stronger the regularization
model_name=params['model_name']
print('Model name is %s' % model_name)
model_type=params['model_type']
print('Model type is %s' % model_type)
if model_type!='logreg':
    # Read gamma SVM hyperparameter
    gamma = float(params['gamma'])
    print('Gamma is %f' % gamma)
    #gamma defines how much influence a single training example has. The larger gamma is, the closer other examples must be to be affected.
ftr_types=params['use_ftrs']
print('Features being used: {}'.format(ftr_types))
if params['ictal_wind']=='small':
    small_ictal_wind=True
else:
    small_ictal_wind=False

# Import list of subjects to use
path_dict=ief.get_path_dict()
model_path=os.path.join(path_dict['szr_ant_root'],'MODELS',model_name)
# if os.path.exists(model_path)==False:
#     os.mkdir(model_path)
# Copy json parameter file to model path
# copyfile(param_fname,os.path.join(model_path,'params.json'))
# metrics_file=os.path.join(model_path,'classification_metrics.npz')
use_subs_df=pd.read_csv(os.path.join(path_dict['szr_ant_root'],'use_subs.txt'),header=None,na_filter=False)
test_sub_list=['NA']
train_subs_list=[]
for sub in use_subs_df.iloc[:,0]:
    if not sub in test_sub_list:
        train_subs_list.append(sub)
print('Training subs: {}'.format(train_subs_list))


# Find out how much data exists to preallocate memory
n_ftr_types=len(ftr_types)
n_dim=0
n_wind=0
n_subs=len(train_subs_list)
n_szrs=np.zeros(n_subs)
mn_preictal_sec=np.zeros(n_subs)
sub_stem_dict=dict()
for type_ct, ftr_type in enumerate(ftr_types):
    print('Checking dimensions of feature type %s' % ftr_type)
    # Figure out how much data there is to preallocate mem
    ftr_type_dir=os.path.join(path_dict['ftrs_root'],ftr_type)

    if type_ct==0:
        # count time windows and create fname_stem_list
        for sub_ct, sub in enumerate(train_subs_list):
            temp_stem_list = []
            ftr_path=os.path.join(ftr_type_dir,sub)
            for f in os.listdir(ftr_path):
                #get file stem
                f_parts = f.split('_')
                f_stem = f_parts[0]+'_'+f_parts[1]+'_'+f_parts[2]
                #fname_stem_list.append(f_stem)
                temp_stem_list.append(f_stem)
                ftr_dict=np.load(os.path.join(ftr_path,f))
                if small_ictal_wind:
                    n_wind+=np.sum(ftr_dict['peri_ictal']>=0)
                else:
                    n_wind += len(ftr_dict['peri_ictal'])
            sub_stem_dict[sub]=temp_stem_list
            n_szrs[sub_ct]=len(temp_stem_list)
    else:
        temp_n_wind=0
        f_ct=0
        # Load file extension
        ext_fname = os.path.join(ftr_type_dir, 'ext.txt')
        with open(ext_fname, 'r') as f_ext:
            ext = f_ext.readline()[:-1]
        print(ext_fname)
        print(ext)
        for sub_ct, sub in enumerate(train_subs_list):
            ftr_path=os.path.join(ftr_type_dir,sub)
            temp_preictal_sec = np.zeros(len(sub_stem_dict[sub]))
            for temp_stem_ct, temp_stem in enumerate(sub_stem_dict[sub]):
                targ_file=os.path.join(ftr_path,temp_stem+ext)
                # if os.path.isfile(targ_file)==False:
                #     print('File not found: %s' % targ_file)
                #     raise ValueError('File stems do not match across features')
                print('Loading file %s' % targ_file)
                ftr_dict=np.load(targ_file)
                if small_ictal_wind:
                    temp_n_wind+=np.sum(ftr_dict['peri_ictal']>=0)
                else:
                    temp_n_wind += len(ftr_dict['peri_ictal'])
                if type_ct == 1:
                    temp_preictal_sec[temp_stem_ct]=np.max(ftr_dict['time_wind_sec'][ftr_dict['peri_ictal'] == 0])
                f_ct+=1
            if type_ct == 1:
                mn_preictal_sec[sub_ct]=np.mean(temp_preictal_sec)
        if temp_n_wind!=n_wind:
            raise ValueError('# of time windows do not match across features')
    n_dim += ftr_dict['ftrs'].shape[0]

print('Total # of dimensions: %d ' % n_dim)
print('Total # of time windows: %d ' % n_wind)
print('Total # of files: %d' % f_ct)
print('Mean [SD] # of szrs: %f [%f]' % (np.mean(n_szrs),np.std(n_szrs)))
print(mn_preictal_sec)
print('Mean [SD] # sec of preictal dta: %f [%f]' % (np.mean(mn_preictal_sec),np.std(mn_preictal_sec)))