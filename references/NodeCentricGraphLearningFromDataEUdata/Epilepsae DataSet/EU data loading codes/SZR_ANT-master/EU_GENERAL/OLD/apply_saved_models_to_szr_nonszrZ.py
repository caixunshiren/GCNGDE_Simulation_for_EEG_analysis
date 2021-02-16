# This script applies a saved ensemble of models to all of a subject's szr files and outputs the
# mean predictions to a mat file in the model's directory. This is useful for figuring out how well the
# model is at detecting ictal activity.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
import os
import sys
import ieeg_funcs as ief
import dgFuncs as dg
import pickle
from sklearn import svm
from sklearn.externals import joblib


# Function for extracting channel names from filename
def chan_labels_from_fname(in_file):
    just_fname=in_file.split('/')[-1]
    jf_splt=just_fname.split('_')
    chan_label=jf_splt[1]+'-'+jf_splt[2]
    return chan_label


# Get list of electrodes for the subject
def sub_soz_elec_names(sub, ftr_root):
    soz_elec_names = list()
    szr_fname_dict = dict()
    non_elec_names = list()

    ftr_path = os.path.join(ftr_root, str(sub))
    for f in os.listdir(ftr_path):
        if f.endswith('non.mat'):
            non_elec_names.append(chan_labels_from_fname(f))
        elif f.endswith('.mat') and f.startswith(str(sub) + '_'):
            temp_label = chan_labels_from_fname(f)
            if temp_label in soz_elec_names:
                szr_fname_dict[soz_elec_names[-1]].append(f)
            else:
                soz_elec_names.append(temp_label)
                szr_fname_dict[temp_label] = [f]

    soz_elec_names = np.unique(soz_elec_names)
    non_elec_names = np.unique(non_elec_names)
    print('%d total # of electrodes for this sub' % len(soz_elec_names))

    return soz_elec_names, szr_fname_dict



## Start of main function
if len(sys.argv)==1:
    print('Usage: apply_saved_models_to_szr.py sub_number model_name')
    exit()
if len(sys.argv)!=3:
    raise Exception('Error: apply_saved_models_to_szr.py requires 2 arguments: sub_number model_name')

# Import Parameters from json file
sub=int(sys.argv[1])
model_name=sys.argv[2]

path_dict=ief.get_path_dict()
# print(path_dict.keys())
# print(path_dict['szr_ant_root'])
# exit()

# Load models
# model_stem='genLogregSe_3'
model_stem=model_name.split('_')[0]
# TODO get this from pathdir
model_fname=os.path.join(path_dict['szr_ant_root'],'MODELS/',model_name,'classify_models_srch.pkl')
print('Loading models from %s' % model_fname)
models=pickle.load(open(model_fname,'rb'))
n_models=len(models)
print('# of models= %d' % n_models)

# TODO get this from pathdir
yhat_dir = os.path.join(path_dict['szr_ant_root'],'MODELS/', model_name + '_yhat')
if os.path.exists(yhat_dir)==False:
    os.mkdir(yhat_dir)
ftr_root = os.path.join(path_dict['eu_gen_ftrs'],'SE') # TODO make able to deal with multiple features
# Get list of electrodes
soz_elec_names, szr_fname_dict = sub_soz_elec_names(sub, ftr_root)
ftr_path = os.path.join(ftr_root, str(sub))
for elec in soz_elec_names:
    # load non szr file
    uni_chans = elec.split('-')
    nonszr_fname = str(sub) + '_' + uni_chans[0] + '_' + uni_chans[1] + '_non.mat'
    # print('Loading %s' % nonszr_fname)
    temp_ftrs = sio.loadmat(os.path.join(ftr_path, nonszr_fname))
    # Z-score features
    raw_ftrs=temp_ftrs['nonszr_se_ftrs']
    temp_mns, temp_sds = dg.trimmed_normalize(raw_ftrs, 0, zero_nans=False, verbose=False)

    # Apply classifier to non-szr data
    for model_ct in range(n_models):
        tmp_yhat_va = models[model_ct].predict_proba(raw_ftrs.T)[:, 1]
        if model_ct == 0:
            yhat = np.zeros(tmp_yhat_va.shape)
        yhat += tmp_yhat_va / n_models
    out_fname = str(sub) + '_' + uni_chans[0] + '_' + uni_chans[1] + '_phat_non.mat'
    print('Saving file as %s' % os.path.join(yhat_dir, out_fname))
    sio.savemat(os.path.join(yhat_dir, out_fname), mdict={'yhat': yhat,
                                                          'model_name': model_name,
                                                          'ftrs_z': raw_ftrs,
                                                          'ftr_labels': temp_ftrs['ftr_labels'],
                                                          'ftr_fname': nonszr_fname})

    # get list of szr files
    for szr_f in szr_fname_dict[elec]:
        # load file
        #         print('Loading %s' % szr_f)
        temp_ftrs = sio.loadmat(os.path.join(ftr_path, szr_f))
        raw_ftrs = temp_ftrs['se_ftrs']
        # Z-score based on non-ictal means, SDs
        dg.applyNormalize(raw_ftrs, temp_mns, temp_sds)

        # Apply classifiers
        for model_ct in range(n_models):
            tmp_yhat_va = models[model_ct].predict_proba(raw_ftrs.T)[:, 1]
            if model_ct == 0:
                yhat = np.zeros(tmp_yhat_va.shape)
            yhat += tmp_yhat_va / n_models
        out_fname = str(sub) + '_' + uni_chans[0] + '_' + uni_chans[1] + '_phat_' + szr_f.split('_')[-1]
        print('Saving file as %s' % os.path.join(yhat_dir, out_fname))
        sio.savemat(os.path.join(yhat_dir, out_fname), mdict={'yhat': yhat,
                                                              'model_name': model_name,
                                                              'ftrs_z': raw_ftrs,
                                                              'ftr_labels': temp_ftrs['ftr_labels'],
                                                              'ftr_fname': szr_f})

print('Done!')