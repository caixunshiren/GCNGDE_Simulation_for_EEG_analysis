""" This script loads all the training data of a particular type of feature (e.g., EU_MAG_LAG0) from a subject and
computes the trimmed mean and standard deviation of the feature across all NON-ictal time windows. These are then saved to
disk for feature normalization in the feature directory with the filename ftr_nrms.npz. 70% of the data are used when
computing mean and SD. """
import numpy as np
import os
import sys
import pickle
import ieeg_funcs as ief
import dgFuncs as dg

if len(sys.argv)==1:
    print('Usage: compute_ftr_mn_sd.py patient_id ftr_name')
    exit()
if len(sys.argv)!=3:
    raise Exception('Error: compute_ftr_mn_sd.py requires 2 arguments: patient_id ftr_name (e.g., python compute_ftr_mn_sd.py 1096 EU_MAG_LAG0)')

sub=sys.argv[1]
print('Patient being processed is %s' % sub)
ftr_name=sys.argv[2]
print('Feature being processed is %s' % ftr_name)

edge_pts=1177 # # of time pts at the start of each file to ignore due to edge effects
trim_pptn=0.15 # Not clear if this makes a difference
#trim_pptn=0

# Get key directories
dir_dict=ief.get_path_dict()
print(dir_dict.keys())
print(dir_dict['ftrs_root'])
meta_dir=dir_dict['eu_meta']


# Get list of files to be used for training
split_fname=os.path.join(meta_dir,'data_splits_FR_'+sub+'.pkl')
print('Loading %s' % split_fname)
split_dict=pickle.load(open(split_fname,'rb'))
split_dict.keys()
train_files=split_dict['train_files']


# Get list of feature files
ftr_dir=os.path.join(dir_dict['ftrs_root'],ftr_name,sub)
class_dir=os.path.join(dir_dict['ftrs_root'],'EU_SZR_CLASS',sub)
file_stems=list()
ftr_fnames=list()
for ftr_fname in os.listdir(ftr_dir):
    if ftr_fname.endswith('.npz'):
        tmp=ftr_fname.split('_')
        tmp_stem=tmp[0]+'_'+tmp[1]
        if tmp_stem in train_files:
            file_stems.append(tmp_stem)
            ftr_fnames.append(ftr_fname)
print('%d training files' % len(file_stems))


# TODO Load one file to get the # of features
ftr_dict=np.load(os.path.join(ftr_dir,ftr_fnames[0]))
n_ftr=ftr_dict['ftrs'].shape[0]
print('n_ftr=%d' % n_ftr)

# Compute mean and sd of non-ictal points in training data
n_files=len(ftr_fnames)
mns=np.zeros((n_ftr,n_files))
sds=np.zeros((n_ftr,n_files))
for f_ct, f_stem in enumerate(file_stems):
    # Load data
    ftr_dict=np.load(os.path.join(ftr_dir,ftr_fnames[f_ct]))
    #ftr_dict contains ['ftrs', 'time_wind_sec', 'ftr_list']
    
    if f_ct==0:
        # Get copy of feature labels to save with feature norms
        ftr_list=ftr_dict['ftr_list']
    
    # Load szr classes
    class_dict=np.load(os.path.join(class_dir,f_stem+'_szr_class.npz'))
    ignore_tpts=np.copy(class_dict['szr_class'])
    ignore_tpts[:edge_pts]=1 #ignore initial time points that are corrupted by edge effects
    #Compute trimmed mean and sd (70% of data will be used for estimating mean & SD)
    if trim_pptn == 0:
        # Compute regular mean and SD
        mns[:, f_ct] = np.mean(ftr_dict['ftrs'][:, ignore_tpts==0], axis=1)
        sds[:, f_ct] = np.std(ftr_dict['ftrs'][:, ignore_tpts==0], axis=1)
    else:
        for ftr_ct in range(n_ftr):
            mns[ftr_ct,f_ct], sds[ftr_ct,f_ct]=dg.trimmed_mn_sd(ftr_dict['ftrs'][ftr_ct, ignore_tpts==0],trim_pptn)

# Get trimmed mean and SD (70% of data will be used for estimating mean & SD)
nrm_mn=np.zeros(n_ftr)
nrm_sd=np.zeros(n_ftr)
for ftr_ct in range(n_ftr):
    if trim_pptn==0:
        nrm_mn[ftr_ct]= np.mean(mns[ftr_ct, :])
        nrm_sd[ftr_ct] = np.mean(sds[ftr_ct, :])
    else:
        nrm_mn[ftr_ct], _=dg.trimmed_mn_sd(mns[ftr_ct,:],trim_pptn)
        nrm_sd[ftr_ct], _=dg.trimmed_mn_sd(sds[ftr_ct,:],trim_pptn)
        #nrm_mn[ftr_ct], nrm_sd[ftr_ct]=dg.trimmed_mn_sd(mns[ftr_ct,:],trim_pptn)


#SAVE nrm_mn and nrm_sd to disk for data preprocessing
out_fname=os.path.join(ftr_dir,'ftr_nrms.npz')
print('Saving feature norms to %s' % out_fname)
np.savez(out_fname, nrm_mn=nrm_mn, nrm_sd=nrm_sd, ftr_list=ftr_list, file_stems=file_stems)

