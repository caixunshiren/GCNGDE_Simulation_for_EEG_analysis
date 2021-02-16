# This script applies a saved ensemble of models to all of a subject's continuous iEEG files and outputs the
# mean predictions to one of these places
# if sys.platform=='linux':
#     out_root = '/home/dgroppe/EU_YHAT/'
# else:
#     out_root = '/Users/davidgroppe/ONGOING/EU_YHAT/'

# Libraries
import numpy as np
import scipy.io as sio
import os
import sys
import pickle
import ieeg_funcs as ief
import dgFuncs as dg
# from sklearn import preprocessing
from sklearn import svm, linear_model
# from sklearn.metrics import roc_auc_score
# from sklearn.externals import joblib
import matplotlib.pyplot as plt

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
    print('%d total # of SOZ electrodes for this sub' % len(soz_elec_names))

    return soz_elec_names, szr_fname_dict


# Get filenames grouped by time chunk
def continuous_fnames(sub, cont_ftr_root, szr_ftr_root):
    # /home/dgroppe/EU_SE_FTRS/1096_all

    # Get list of SOZ electrodes
    soz_elec_names, _ = sub_soz_elec_names(sub, szr_ftr_root)
    first_chans = soz_elec_names[0].split('-')

    # Get list of all time clips
    ftr_path = os.path.join(cont_ftr_root, str(sub) + '_all')
    clip_list = list()
    for f in os.listdir(ftr_path):
        if f.startswith(str(sub) + '_' + first_chans[0] + '_' + first_chans[1]):
            stem = f.split('.')[0]
            splt_stem = stem.split('_')
            clip_list.append(splt_stem[3] + '_' + splt_stem[4])

    # Loop over soz electrodes and make sure time chunk exists for each electrode
    flag = False
    for chan in soz_elec_names[1:]:  # skip first electrode because we know files exist
        mono_chans = chan.split('-')
        for clip in clip_list:
            fname = str(sub) + '_' + mono_chans[0] + '_' + mono_chans[1] + '_' + clip + '.mat'
            f = os.path.join(ftr_path, fname)
            if os.path.isfile(f) == False:
                print('Error: missing file %s' % f)
                flag = True
    if flag == False:
        print('All files accounted for. Proceed.')

    return clip_list, soz_elec_names



# Get list of files
sub=1096
## Start of main function
if len(sys.argv)==1:
    print('Usage: apply_saved_models_to_all_clips.py sub_id model_name')
    exit()
if len(sys.argv)!=3:
    raise Exception('Error: apply_saved_models_to_all_clips.py requires 2 arguments: sub_id model_name')

# Import Parameters from command line
sub = int(sys.argv[1])
model_name=sys.argv[2]

# TODO fix
path_dict=ief.get_path_dict()
if sys.platform=='linux':
    cont_ftr_root='/home/dgroppe/EU_SE_FTRS/' # location of features for continuous data
else:
    cont_ftr_root='/Users/davidgroppe/ONGOING/EU_SE_FTRS/'
szr_ftr_root = os.path.join(path_dict['eu_gen_ftrs'],'SE') # location of subsampled features

# Get list of continuous feature files (most each about an hour long)
clip_list, soz_elec_names=continuous_fnames(sub, cont_ftr_root, szr_ftr_root)
n_chan=len(soz_elec_names)
print('# of clips %d' % len(clip_list))

# Load models
# model_name='genSvmSe_3'
# model_name = 'genLogregSe_3'
# model_type='svm'
model_type = 'logreg' # TODO make this work for SVMS too
# model_fname=os.path.join('/home/dgroppe/GIT/SZR_ANT/MODELS/',model_name,'classify_models_srch.pkl')

# TODO fix path
model_fname = os.path.join(path_dict['szr_ant_root'],'MODELS', model_name, 'classify_models_srch.pkl')
models = pickle.load(open(model_fname, 'rb'))
n_models = len(models)
print('# of models= %d' % n_models)

# tpt span of moving window via which to smooth p(stim)
mv_wind_len = 20  # Sampling rate is about 10 Hz, so this a 2 sec moving window

# Outpath #TODO fix
if sys.platform=='linux':
    out_root = '/home/dgroppe/EU_YHAT/'
else:
    out_root = '/Users/davidgroppe/ONGOING/EU_YHAT/'
out_path = os.path.join(out_root, str(sub) + '_' + model_name)
if os.path.exists(out_path) == False:
    os.mkdir(out_path)

# Load params for normalization
subsamp_ftr_root = os.path.join(path_dict['szr_ant_root'],'EU_GENERAL','EU_GENERAL_FTRS','SE')
# subsamp_ftr_root = '/Users/davidgroppe/PycharmProjects/SZR_ANT/EU_GENERAL/EU_GENERAL_FTRS/SE/'
# subsamp_ftr_root='/home/dgroppe/GIT/SZR_ANT/EU_GENERAL/EU_GENERAL_FTRS/SE/'
subsamp_ftr_path = os.path.join(subsamp_ftr_root, str(sub))
mns_dict = dict()
sds_dict = dict()
for chan in soz_elec_names:
    mono_chans = chan.split('-')
    nonszr_fname = str(sub) + '_' + mono_chans[0] + '_' + mono_chans[1] + '_non.mat'
    # print('Loading %s' %
    temp_ftrs = sio.loadmat(os.path.join(subsamp_ftr_path, nonszr_fname))
    # Z-score features
    temp_mns, temp_sds = dg.trimmed_normalize(temp_ftrs['nonszr_se_ftrs'], 0,
                                              zero_nans=False, verbose=False)
    mns_dict[chan] = temp_mns
    sds_dict[chan] = temp_sds

# Loop over clips
# for clip in clip_list[:1]:
for clip in clip_list:
    print('Working on clip %s' % clip)
    # Loop over SOZ electrodes
    for chan_ct, chan in enumerate(soz_elec_names):
        mono_chans = chan.split('-')
        # Load features
        fname = str(sub) + '_' + mono_chans[0] + '_' + mono_chans[1] + '_' + clip + '.mat'
        f = os.path.join(cont_ftr_root, str(sub) + '_all', fname)
        temp_mat = sio.loadmat(f)

        # Estimate p(szr)
        raw_ftrs = temp_mat['se_ftrs']
        # Z-score based on non-ictal means, SDs
        dg.applyNormalize(raw_ftrs, mns_dict[chan], sds_dict[chan])
        # Apply classifiers
        for model_ct in range(n_models):
            if model_type == 'svm':
                tmp_yhat_va = models[model_ct].predict(raw_ftrs.T)[:, 1]
            else:
                tmp_yhat_va = models[model_ct].predict_proba(raw_ftrs.T)[:, 1]
            if model_ct == 0:
                yhat = np.zeros(tmp_yhat_va.shape)
            yhat += tmp_yhat_va / n_models

        # Smooth p(szr)
        yhat_smooth = np.zeros(yhat.shape)
        yhat_smooth[mv_wind_len - 1:] = dg.running_mean(yhat, mv_wind_len)

        # Collect max(p(szr)) for each time window
        if chan_ct == 0:
            # first channel
            max_yhat = np.copy(yhat_smooth)
            yhat_soz_chans = np.zeros((n_chan, len(yhat_smooth)))
        else:
            max_yhat = np.maximum(max_yhat, yhat_smooth)
        yhat_soz_chans[chan_ct, :] = yhat_smooth

    # Write to disk
    out_fname = clip + '_yhat'
    print('Saving yhat to %s' % os.path.join(out_path, out_fname))
    np.savez(os.path.join(out_path, out_fname),
             max_yhat=max_yhat,
             yhat_soz_chans=yhat_soz_chans,
             yhat_sec=temp_mat['se_time_sec'])  # note that time is relative to start of file (i.e., 0=first file tpt)

print('Done.')