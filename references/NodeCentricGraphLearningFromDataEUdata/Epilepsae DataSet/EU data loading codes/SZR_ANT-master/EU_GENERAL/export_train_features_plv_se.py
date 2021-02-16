# Performs LOOCV on train_subs.txt using a grid search of gamma values and hand set initial C value. You need to
# set these yourself in the code (it is not in the json parameter file).
# Some parameters are fed in via a json file like this:
# {"model_type": "svm",
# "model_name": "gen_marr_svm_se",
# "gam": 0.01,
# "patience": 0,
# "use_ftrs": ["SE"]}
#
# TODO: currently code only works with one feature. Need to add others.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
import os
import sys
import ieeg_funcs as ief
import dgFuncs as dg
import euGenFuncs as eu
import pickle
from sklearn import svm
from sklearn.externals import joblib
import json


## Start of main function
# if len(sys.argv)==1:
#     print('Usage: train_smart_srch_multi_patient.py srch_params.json')
#     exit()
# if len(sys.argv)!=2:
#     raise Exception('Error: train_smart_srch_multi_patient.py requires 1 argument: srch_params.json')

# Import Parameters from json file
# param_fname=sys.argv[1]
# print('Importing model parameters from %s' % param_fname)
# with open(param_fname) as param_file:
#     params=json.load(param_file)
# model_name=params['model_name']
# print('Model name is %s' % model_name)
# model_type=params['model_type']
# print('Model type is %s' % model_type)
# ftr_types=params['use_ftrs']
# print('Features being used: {}'.format(ftr_types))
# if params['equal_sub_wts']=="False":
#     equal_sub_wts=False
# else:
#     equal_sub_wts = True
# print('Weight subjects equally={}'.format(equal_sub_wts))

# if params['ictal_wind']=='small':
#     small_ictal_wind=True
# elif params['ictal_wind']=='max':
#     small_ictal_wind=False
# else:
#     raise Exception('ictal_wind needs to be "small" or "max"')
ftr='PLV_SE' #TODO import this from json file
bnded=True
# n_rand_params=int(params['n_rand_params'])
# print('# of random initial hyperparameters to try %d' % n_rand_params)
# n_rand_params=1
# gamma_vals=params['gam']
# print('Gamma value: %f' % gamma_vals)
# ini_C=params['C']
# print('Initial C value: %f' % ini_C)
# patience=int(params['patience'])
# print('# of steps to wait when performance no longer increases %d' % patience)

# Find if there any existing models of this name
# If so, grab the number of model an increment that number by 1 to get new model name
path_dict=ief.get_path_dict()
# model_root=model_path=os.path.join(path_dict['szr_ant_root'],'MODELS')
# model_num=1
# for f in os.listdir(model_root):
#     if os.path.isdir(os.path.join(model_root,f)):
#         spltf=f.split('_')
#         if spltf[0]==model_name:
#            temp_model_num=int(spltf[1])
#            if temp_model_num>=model_num:
#                model_num=temp_model_num+1
# model_path=os.path.join(model_root,model_name+'_'+str(model_num))
# print('Model will be stored to %s' % model_path)
# if os.path.exists(model_path)==False:
#     os.mkdir(model_path)

# Import list of subjects to use
#use_subs_df=pd.read_csv(os.path.join(path_dict['szr_ant_root'],'use_subs.txt'),header=None,na_filter=False)
train_sub_fname='train_subsAES.txt' #TODO import this from cmnd line
print('Importing training subject list: %s' % train_sub_fname)
use_subs_df=pd.read_csv(train_sub_fname,header=None,na_filter=False)
train_subs_list=[]
for sub in use_subs_df.iloc[:,0]:
    train_subs_list.append(sub)
        
print('Training subs: {}'.format(train_subs_list))


# ftr_root='/Users/davidgroppe/PycharmProjects/SZR_ANT/EU_GENERAL/EU_GENERAL_FTRS/SE/'
ftr_root=path_dict['eu_gen_ftrs']
ftr_info_dict=eu.data_size_and_fnames(train_subs_list, ftr_root, ftr)
#ftrs_tr, targ_labels_tr=eu.import_data(szr_fnames_tr, non_fnames_tr, n_szr_wind_tr, n_non_wind_tr, ftr_dim)

n_dim=ftr_info_dict['ftr_dim']
n_non_wind=ftr_info_dict['grand_n_non_wind']
n_szr_wind=ftr_info_dict['grand_n_szr_wind']
n_wind=n_non_wind+n_szr_wind
print('Total # of dimensions: %d ' % n_dim)
print('Total # of szr time windows: %d ' % n_szr_wind)
print('Total # of non-szr time windows: %d ' % n_non_wind)
print('Total # of time windows: %d ' % n_wind)
# print('Total # of files: %d' % f_ct)

if n_dim==0:
    print('Error: ftr_dim=0. You might be missing some *non* or *szr* files for a patient')
    exit()

# Load training/validation data into a single matrix
ftrs, szr_class, sub_id=eu.import_data(ftr_info_dict['grand_szr_fnames'], ftr_info_dict['grand_non_fnames'],
                                       ftr_info_dict['szr_file_subs'],ftr_info_dict['non_file_subs'],
                                       n_szr_wind, n_non_wind, n_dim, dsamp_pcnt=1, bnded=bnded)

if bnded==True:
    out_fname='train_ftrs_bnded.npz'
else:
    out_fname='train_ftrs_unbnded.npz'
print('Saving training features (and associated metadata) to %s' % out_fname)
np.savez(out_fname,ftrs=ftrs,szr_class=szr_class,sub_id=sub_id,train_subs_list=train_subs_list,bnded=bnded)