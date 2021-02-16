# Function for applying a saved ensemble of models to subsampled data from a list of subjects (sub_list.txt)
# These are the data that could be used for training
# It outputs binary classification performance metrics

import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
# import scipy.io as sio
import os
import sys
import ieeg_funcs as ief
import dgFuncs as dg
import euGenFuncs as eu
import pickle
from sklearn.metrics import roc_auc_score
from sklearn import svm
from sklearn.externals import joblib
import json


## Start of main function
if len(sys.argv)==1:
    print('Usage: apply_models_to_subsamp.py model_name sub_list.txt ftr')
    exit()
if len(sys.argv)!=4:
    raise Exception('Error: apply_models_to_subsamp.py requires 3 arguments: model_name sub_list.txt ftr')

# Import Parameters from command line
model_name=sys.argv[1]
print('Model name is %s' % model_name)

text_file=open(sys.argv[2],'r')
temp=text_file.readlines()
subs=list()
for raw_sub in temp:
    subs.append(int(raw_sub.strip()))
print('Subs are {}'.format(subs))
path_dict=ief.get_path_dict()

ftr=sys.argv[3]
print('Features are %s' % ftr)

# Load model
model_root=model_path=os.path.join(path_dict['szr_ant_root'],'MODELS')
model_fname=os.path.join(model_root,model_name,'classify_models_srch.pkl')
print('Loading %s' % model_fname)
models=pickle.load(open(model_fname,'rb'))
n_models=len(models)
print('# of models in ensemble= %d' % n_models)

sub_type=sys.argv[2].split('_')[0]+'.npz'
out_fname=os.path.join(model_root,model_name,'apply_models_to_subsamp_'+sub_type)
print('Saving results to %s' % out_fname)

# Import Data
# use_ftrs=['SE'] #TODO import this from model
ftr_root=path_dict['eu_gen_ftrs']
#ftr='SE'
ftr_info_dict=eu.data_size_and_fnames(subs, ftr_root, ftr)

n_dim=ftr_info_dict['ftr_dim']
n_non_wind=ftr_info_dict['grand_n_non_wind']
n_szr_wind=ftr_info_dict['grand_n_szr_wind']
n_wind=n_non_wind+n_szr_wind
print('Total # of dimensions: %d ' % n_dim)
print('Total # of szr time windows: %d ' % n_szr_wind)
print('Total # of non-szr time windows: %d ' % n_non_wind)
print('Total # of time windows: %d ' % n_wind)
# print('Total # of files: %d' % f_ct)


# Load training/validation data into a single matrix
ftrs, szr_class, sub_id=eu.import_data(ftr_info_dict['grand_szr_fnames'], ftr_info_dict['grand_non_fnames'],
                                       ftr_info_dict['szr_file_subs'],ftr_info_dict['non_file_subs'],
                                       n_szr_wind, n_non_wind, n_dim)

#print('Max ftrs %f' % np.max(ftrs))

# TODO rm np.savez('tempftrs.npz',ftrs=ftrs)
# Set sample weights to weight each subject (and preictal/ictal equally:
uni_subs = np.unique(sub_id)
n_subs = len(uni_subs)
wt_subs_equally=False
# if wt_subs_equally==True:
#     samp_wts = np.ones(n_wind)
#     for sub_loop in uni_subs:
#         subset_id = (sub_id == sub_loop)
#         n_obs = np.sum(subset_id)
#         print('Sub #%d has %d observations' % (sub_loop, int(n_obs)))
#         samp_wts[subset_id] = samp_wts[subset_id] / n_obs
#     print('Sum samp wts=%f' % np.sum(samp_wts))
#     print('# of subs=%d' % n_subs)


# Apply ensemble of models on validation data
for model_ct in range(n_models):
    print('Working on model %d of %d' % (model_ct+1,n_models))
    tmp_yhat = models[model_ct].predict(ftrs)
    print(tmp_yhat.shape)
    #tmp_yhat = models[model_ct].predict(ftrs)[:, 1]
    if model_ct == 0:
        class_hat = np.zeros(tmp_yhat.shape)
    class_hat += tmp_yhat / n_models


# Compute performance metrics
print('Ensemble Performance (each obs wtd equally)')
auc = roc_auc_score(szr_class, class_hat)
print('AUC=%.3f' % auc)
bal_acc, sens, spec = ief.perf_msrs(szr_class, class_hat >= 0.5)
print('Balanced Accuracy (sens/spec)=%.3f (%f/%f)' % (bal_acc, sens, spec))

# Report results for individual subjects
print('Subject specific performance')
crct=(class_hat>0.5)==szr_class
ictal=szr_class==1
bacc_by_sub=np.zeros(n_subs)
sens_by_sub=np.zeros(n_subs)
spec_by_sub=np.zeros(n_subs)
auc_by_sub=np.zeros(n_subs)
for sub_ct, sub in enumerate(subs):
    print()
    sub_bool=sub_id==sub
    print('Sub %d' % int(sub))
    print('%f of data' % np.mean(sub_bool))
    #sens=np.mean(crct[ictal and sub_bool])
    sens_by_sub[sub_ct]=np.mean(crct[np.multiply(ictal,sub_bool)])
    spec_by_sub[sub_ct]=np.mean(crct[np.multiply(ictal==False,sub_bool)])
    bacc_by_sub[sub_ct]=(sens_by_sub[sub_ct]+spec_by_sub[sub_ct])/2
    print('Acc (Sens/Spec): %f (%f/%f)' % (bacc_by_sub[sub_ct],
                                           sens_by_sub[sub_ct],
                                           spec_by_sub[sub_ct]))
    auc_by_sub[sub_ct]=roc_auc_score(szr_class[sub_bool], class_hat[sub_bool])
    print('AUC=%f' % auc_by_sub[sub_ct])

print('Mean across subs:')
m, low, hi=dg.mean_and_cis(bacc_by_sub)
print('Mean (95p CI) bacc: %f (%f-%f)' % (m,low,hi))
m, low, hi=dg.mean_and_cis(spec_by_sub)
print('Mean (95p CI) spec: %f (%f-%f)' % (m,low,hi))
m, low, hi=dg.mean_and_cis(sens_by_sub)
print('Mean (95p CI) sens: %f (%f-%f)' % (m,low,hi))
m, low, hi=dg.mean_and_cis(auc_by_sub)
print('Mean (95p CI) AUC: %f (%f-%f)' % (m,low,hi))

np.savez(out_fname,
         subs=subs,
         n_models=n_models,
         bacc_by_sub=bacc_by_sub,
         spec_by_sub=spec_by_sub,
         sens_by_sub=sens_by_sub,
         auc_by_sub=auc_by_sub)
