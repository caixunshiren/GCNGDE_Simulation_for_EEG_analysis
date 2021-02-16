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
from datetime import datetime


## Start of main function
if len(sys.argv)==1:
    print('Usage: train_smart_srch_multi_patient.py srch_params.json')
    exit()
if len(sys.argv)!=2:
    raise Exception('Error: train_smart_srch_multi_patient.py requires 1 argument: srch_params.json')

# Import Parameters from json file
param_fname=sys.argv[1]
print('Importing model parameters from %s' % param_fname)
with open(param_fname) as param_file:
    params=json.load(param_file)
model_name=params['model_name']
print('Model name is %s' % model_name)
model_type=params['model_type']
print('Model type is %s' % model_type)
ftr_types=params['use_ftrs']
print('Features being used: {}'.format(ftr_types))
if params['equal_sub_wts']=="False":
    equal_sub_wts=False
else:
    equal_sub_wts = True
print('Weight subjects equally={}'.format(equal_sub_wts))
# if params['ictal_wind']=='small':
#     small_ictal_wind=True
# elif params['ictal_wind']=='max':
#     small_ictal_wind=False
# else:
#     raise Exception('ictal_wind needs to be "small" or "max"')
use_ftrs=['SE'] #TODO import this from json file
# n_rand_params=int(params['n_rand_params'])
# print('# of random initial hyperparameters to try %d' % n_rand_params)
n_rand_params=1
gamma_vals=params['gam']
print('Gamma value: %f' % gamma_vals)
ini_C=params['C']
print('Initial C value: %f' % ini_C)
patience=int(params['patience'])
print('# of steps to wait when performance no longer increases %d' % patience)

# Find if there any existing models of this name
# If so, grab the number of model an increment that number by 1 to get new model name
path_dict=ief.get_path_dict()
model_root=model_path=os.path.join(path_dict['szr_ant_root'],'MODELS')
model_num=1
for f in os.listdir(model_root):
    if os.path.isdir(os.path.join(model_root,f)):
        spltf=f.split('_')
        if spltf[0]==model_name:
           temp_model_num=int(spltf[1])
           if temp_model_num>=model_num:
               model_num=temp_model_num+1
model_path=os.path.join(model_root,model_name+'_'+str(model_num))
print('Model will be stored to %s' % model_path)
if os.path.exists(model_path)==False:
    os.mkdir(model_path)

# Import list of subjects to use
#use_subs_df=pd.read_csv(os.path.join(path_dict['szr_ant_root'],'use_subs.txt'),header=None,na_filter=False)
use_subs_df=pd.read_csv('train_subs.txt',header=None,na_filter=False)
#test_sub_list=['NA']
test_sub_list=['1096']
train_subs_list=[]
for sub in use_subs_df.iloc[:,0]:
    if not sub in test_sub_list:
        train_subs_list.append(sub)
        
print('Training subs: {}'.format(train_subs_list))


# ftr_root='/Users/davidgroppe/PycharmProjects/SZR_ANT/EU_GENERAL/EU_GENERAL_FTRS/SE/'
ftr_root=path_dict['eu_gen_ftrs']
ftr='SE'
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

# Load training/validation data into a single matrix
ftrs, szr_class, sub_id=eu.import_data(ftr_info_dict['grand_szr_fnames'], ftr_info_dict['grand_non_fnames'],
                                       ftr_info_dict['szr_file_subs'],ftr_info_dict['non_file_subs'],
                                       n_szr_wind, n_non_wind, n_dim)
# Downsample data
dsamp_fact=10
print('Downsample factor %d !!!!!!' % dsamp_fact)
# print(ftrs.shape)
ftrs=ftrs[0::dsamp_fact]
szr_class=szr_class[0::dsamp_fact]
sub_id=sub_id[0::dsamp_fact]
n_wind=len(szr_class)
print('Total # of DOWNSAMPLED time windows: %d ' % n_wind)

# Set sample weights to weight each subject (and preictal/ictal equally:
uni_subs=np.unique(sub_id)
n_train_subs = len(uni_subs)
samp_wts = np.ones(n_wind)
for sub_loop in uni_subs:
    subset_id = (sub_id == sub_loop)
    n_obs = np.sum(subset_id)
    print('Sub #%d has %d observations' % (sub_loop, int(n_obs)))
    samp_wts[subset_id] = samp_wts[subset_id] / n_obs
print('Sum samp wts=%f' % np.sum(samp_wts))
print('# of subs=%d' % n_train_subs)

#np.savez('temp_data.npz',ftrs=ftrs, szr_class=szr_class, sub_id=sub_id)
#exit()
# print('File ct=%d' % file_ct)
# print('wind_ct=%d' % wind_ct)
# np.savez('temp.npz',ftrs=ftrs,szr_class=szr_class,sub_id=sub_id)
# np.savez('temp_ftrs.npz',ftrs=ftrs,szr_class=szr_class,sub_id=sub_id)
# exit()


# LOOCV on training data
valid_sens = np.zeros((n_train_subs,n_rand_params))
valid_spec = np.zeros((n_train_subs,n_rand_params))
valid_acc = np.zeros((n_train_subs,n_rand_params))
valid_bal_acc = np.zeros((n_train_subs,n_rand_params))
train_sens = np.zeros((n_train_subs,n_rand_params))
train_spec = np.zeros((n_train_subs,n_rand_params))
train_acc = np.zeros((n_train_subs,n_rand_params))
train_bal_acc = np.zeros((n_train_subs,n_rand_params))
pptn_missed_szrs = np.zeros((n_train_subs,n_rand_params))
pptn_preonset_stim = np.zeros((n_train_subs,n_rand_params))
mn_stim_latency = np.zeros((n_train_subs,n_rand_params))
n_train_steps=np.zeros(n_rand_params)
#C_vals=np.random.exponential(1,n_rand_params)
#gamma_vals=np.random.exponential(1,n_rand_params)
C_vals=np.ones(n_rand_params)*ini_C # Start at ini_C from json file
# C = SVM regularization parameter, the smaller it is, the stronger the regularization
#gamma_vals=10**np.random.uniform(-3,0,n_rand_params)
#gamma defines how much influence a single training example has. The larger gamma is, the closer other examples must be to be affected.
best_valid_bal_acc=0
best_train_bal_acc=0
best_models=None
best_C=None
best_gam=None
best_valid_bal_acc_by_sub=None

# Variables to keep track of the mean LOOCV accuracy of all gamma and C values tried
tried_C=list()
tried_gamma=list()
tried_train_acc=list()
tried_valid_acc=list()

for rand_ct in range(n_rand_params):
    C=C_vals[rand_ct]  # Start with C=1 and then change it according to train-testing error dif
    C_direction=0
    gam=gamma_vals
    # print('Random run %d/%d' % (rand_ct+1,n_rand_params))
    print('Using gamma value of %.2E' % gam)

    best_vbal_acc_this_gam=0 # best balanced accuracy for this value of gamma
    steps_since_best = 0
    for C_loop in range(10): # Note max # of C values to try is 10
        print('Using C value of %f' % C)
        temp_models=dict()

        #time ensemble training
        start_time=datetime.now()

        # Use LOOCV to estimate accuracy for this value of C and gamma
        temp_train_sens = np.zeros(n_train_subs)
        temp_train_spec = np.zeros(n_train_subs)
        temp_train_bacc = np.zeros(n_train_subs)
        temp_valid_sens = np.zeros(n_train_subs)
        temp_valid_spec = np.zeros(n_train_subs)
        temp_valid_bacc = np.zeros(n_train_subs)
        for left_out_ct, left_out_id in enumerate(uni_subs):
            print('Left out sub %d (FR_%d) of %d' % (left_out_ct+1,left_out_id,n_train_subs))
            left_in_ids=np.setdiff1d(uni_subs,left_out_id)
            #rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(ftrs.T, szr_class)
            if 'model' in locals():
                del model # clear model just in case
            if model_type=='svm':
                from sklearn import svm
                model = svm.SVC(class_weight='balanced', C=C, gamma=gam)
            elif model_type=='lsvm':
                from sklearn import svm
                print('Using gamma value of %f' % gam)
                model = svm.SVC(class_weight='balanced', kernel='linear', C=C, gamma=gam)
            else:
                from sklearn import linear_model
                #model = linear_model.LogisticRegression(class_weight='balanced', C=C, penalty='l1')
                model = linear_model.LogisticRegression(class_weight='balanced', C=C)

            train_bool = sub_id != left_out_id
            if equal_sub_wts==True:
                #model.fit(ftrs[train_bool, :], szr_class[train_bool ], sample_weight=samp_wts[subset_id])
                model.fit(ftrs[train_bool, :], szr_class[train_bool], sample_weight=samp_wts[train_bool])
            else:
                model.fit(ftrs[train_bool , :], szr_class[train_bool])
            #model.fit(ftrs[sub_id == 0, :], szr_class[sub_id == 0]) # min training data to test code

            # Save model from this left out sub
            temp_models[left_out_ct]=model

            # make predictions from training and validation data
            class_hat = model.predict(ftrs)
            # Compute performance on training data
            for temp_sub in left_in_ids:
                sub_bool=sub_id==temp_sub
                temp_bacc, temp_sens, temp_spec= ief.perf_msrs(szr_class[sub_bool], class_hat[sub_bool])
                temp_train_bacc[left_out_ct]+=temp_bacc/(n_train_subs-1)
                temp_train_sens[left_out_ct]+=temp_sens/(n_train_subs-1)
                temp_train_spec[left_out_ct]+=temp_spec/(n_train_subs-1)

            # temp_train_bacc[left_out_ct], temp_train_sens[left_out_ct ], temp_train_spec[left_out_ct ] = ief.perf_msrs(
            #     szr_class[train_bool],
            #     class_hat[train_bool])
            temp_valid_bacc[left_out_ct], temp_valid_sens[left_out_ct], temp_valid_spec[left_out_ct] = ief.perf_msrs(
                szr_class[train_bool==False],
                class_hat[train_bool==False])
            # print('Bal Acc (Train/Valid): %.3f/%3f ' % (temp_train_bacc[left_out_ct ],temp_valid_bacc[left_out_ct]))
            # exit()

        mn_temp_valid_bacc=np.mean(temp_valid_bacc)
        mn_temp_train_bacc = np.mean(temp_train_bacc)

        # Keep track of results for this value of C and gamma
        tried_C.append(C)
        tried_gamma.append(gam)
        tried_train_acc.append(mn_temp_train_bacc)
        tried_valid_acc.append(mn_temp_valid_bacc)

        print('Time elapsed: {} seconds'.format((datetime.now()-start_time).total_seconds()))
        exit()

        if mn_temp_valid_bacc>best_vbal_acc_this_gam:
            best_vbal_acc_this_gam=mn_temp_valid_bacc
            best_models_this_gam=temp_models.copy()
            C_vals[rand_ct]=C #Store current best C value for this gamma value
            best_valid_bal_acc_by_sub=temp_valid_bacc

            #TODO: remove this?
            # out_model_fname = os.path.join(model_path, 'temp_classify_models_srch.pkl')
            # print('Saving best for model for this gamma value as %s' % out_model_fname)
            # pickle.dump(best_models_this_gam, open(out_model_fname, 'wb'))

            print('Best valid acc so far: %.2f for current gamma value' % best_vbal_acc_this_gam)
            # Training Data Results
            # train_acc[:,rand_ct]=np.mean(jive[train_bool]) #TODO remove!
            #print('Training accuracy: %f' % train_acc[left_out_ct,rand_ct])
            train_sens[:,rand_ct]=temp_train_sens
            #print('Training sensitivity: %f' % train_sens[left_out_ct,rand_ct])
            train_spec[:,rand_ct]=temp_train_spec
            #print('Training specificity: %f' % train_spec[left_out_ct,rand_ct])
            train_bal_acc[:,rand_ct]=temp_train_bacc
            # print('Training balanced accuracy: %f' % train_bal_acc[left_out_ct,rand_ct])

            # Validation Data Results
            # valid_acc[left_out_ct,rand_ct]=np.mean(jive[valid_bool]) #TODO remove!
            #print('Validation accuracy: %f' % valid_acc[left_out_ct,rand_ct])
            valid_sens[:,rand_ct]=temp_valid_sens
            #print('Validation sensitivity: %f' % valid_sens[left_out_ct,rand_ct])
            valid_spec[:,rand_ct]=temp_valid_spec
            #print('Validation specificity: %f' % valid_spec[left_out_ct,rand_ct])
            valid_bal_acc[:,rand_ct] = temp_valid_bacc
            #print('Validation balanced accuracy: %f' % valid_bal_acc[left_out_ct,rand_ct])

            steps_since_best=0
        else:
            steps_since_best+=1

        # C_change=np.abs(mn_temp_train_bacc-mn_temp_valid_bacc)*20
        # if C_change<2:
        #     C_change=2
        C_change = 2
        print('C_change=%f' % C_change)
        n_train_steps[rand_ct] += 1

        if steps_since_best>patience:
            break
        elif C_direction==1:
            print('Train Acc=%.2f Valid Acc=%.2f, still increasing C' % (mn_temp_train_bacc, mn_temp_valid_bacc))
            C = C * C_change
        elif C_direction==-1:
            print('Train Acc=%.2f Valid Acc=%.2f, still decreasing C' % (mn_temp_train_bacc, mn_temp_valid_bacc))
            C = C/C_change
        elif mn_temp_train_bacc<mn_temp_valid_bacc:
            C = C/5
        elif (mn_temp_train_bacc*.99)<mn_temp_valid_bacc:
            # First time through, C_direction has not been set yet
            print('Train Acc=%.2f Valid Acc=%.2f, increasing C' % (mn_temp_train_bacc,mn_temp_valid_bacc))
            C=C*C_change
            C_direction=1
        else:
            # First time through, C_direction has not been set yet
            print('Train Acc=%.2f Valid Acc=%.2f, decreasing C' % (mn_temp_train_bacc, mn_temp_valid_bacc))
            C=C/C_change
            C_direction = -1


    # Report mean CI performance
    # print('# of patients=%d' % len(train_subs_list))
    print('DONE WITH RANDOM GAMMA VALUE of %.2E' % gam)
    print('Training Data')
    mn, ci_low, ci_hi=dg.mean_and_cis(train_bal_acc[:,rand_ct])
    print('Mean (0.95 CI) Balanced Accuracy %.3f (%.3f-%.3f)' % (mn,ci_low,ci_hi))
    mn, ci_low, ci_hi=dg.mean_and_cis(train_sens[:,rand_ct])
    print('Mean (0.95 CI) Sensitivity %.3f (%.3f-%.3f)' % (mn,ci_low,ci_hi))
    mn, ci_low, ci_hi=dg.mean_and_cis(train_spec[:,rand_ct])
    print('Mean (0.95 CI) Specificity %.3f (%.3f-%.3f)' % (mn,ci_low,ci_hi))

    # print('Mean (0.95 CI) Sensitivty %.3f (%.3f)' % (np.mean(perf['train_sens']),)
    print('Validation Data')
    mn, ci_low, ci_hi=dg.mean_and_cis(valid_bal_acc[:,rand_ct])
    print('Mean (0.95 CI) Balanced Accuracy %.3f (%.3f-%.3f)' % (mn,ci_low,ci_hi))
    mn, ci_low, ci_hi=dg.mean_and_cis(valid_sens[:,rand_ct])
    print('Mean (0.95 CI) Sensitivity %.3f (%.3f-%.3f)' % (mn,ci_low,ci_hi))
    mn, ci_low, ci_hi=dg.mean_and_cis(valid_spec[:,rand_ct])
    print('Mean (0.95 CI) Specificity %.3f (%.3f-%.3f)' % (mn,ci_low,ci_hi))

    temp_mn_acc=np.mean(valid_bal_acc[:, rand_ct],axis=0)
    if  temp_mn_acc> best_valid_bal_acc:
        best_models = best_models_this_gam.copy()
        out_model_fname = os.path.join(model_path, 'classify_models_srch.pkl')
        print('Saving model as %s' % out_model_fname)
        pickle.dump(best_models, open(out_model_fname, 'wb'))
        #best_C=C
        best_C=C_vals[rand_ct]
        best_gam=gam
        best_valid_bal_acc = temp_mn_acc
        best_train_bal_acc = np.mean(train_bal_acc[:, rand_ct],axis=0)
        print('NEW best accuracy so far: %f' % best_valid_bal_acc)
        print('Using C=%.2E and gam=%.2E' % (best_C,best_gam))
    else:
        print('No improvement')
        print('Best accuracy so far is still: %f' % best_valid_bal_acc)
        print('Using C=%.2E and gam=%.2E' % (best_C,best_gam))

    # Save current performance metrics
    out_metrics_fname=os.path.join(model_path,'classify_metrics_srch.npz')
    np.savez(out_metrics_fname,
         valid_sens=valid_sens,
         valid_spec=valid_spec,
         valid_bal_acc=valid_bal_acc,
         best_valid_bal_acc_by_sub=best_valid_bal_acc_by_sub,
         train_sens=train_sens,
         train_spec=train_spec,
         train_bal_acc=train_bal_acc,
         train_subs_list=train_subs_list,
         mn_stim_latency=mn_stim_latency,
         pptn_missed_szrs=pptn_missed_szrs,
         pptn_preonset_stim=pptn_preonset_stim,
         n_train_steps=n_train_steps,
         rand_ct=rand_ct,
         C_vals=C_vals,
         gamma_vals=gamma_vals,
         tried_C=tried_C,
         tried_gamma=tried_gamma,
         tried_train_acc=tried_train_acc,
         tried_valid_acc=tried_valid_acc,
         best_valid_bal_acc=best_valid_bal_acc,
         best_train_bal_acc=best_train_bal_acc,
         best_C = best_C,
         best_gam=best_gam,
         best_models=best_models,
         equal_sub_wts=equal_sub_wts,
         ftr_types=ftr_types,
         left_out_id=left_out_id)

print('Done!')
print('Best training data accuracy: %f' % best_train_bal_acc)
print('Best validation accuracy: %f' % best_valid_bal_acc)
print('Using C=%.2E and gam=%.2E' % (best_C,best_gam))
print('Model name: {}'.format(model_name))
print('Features used: {}'.format(use_ftrs))
print('Metrics save to: %s' % out_metrics_fname)
print('Equal subject wts={}'.format(equal_sub_wts))