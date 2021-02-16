""" This function plots the balanced accuracy on training and validation data as a function of C & gamma
hyperparameters for all models with the same stem name"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import ieeg_funcs as ief
import sys

## Start of main function
if len(sys.argv)==1:
    print('Usage: plt_hyper_srch.py model_stem max_n_svs')
    exit()
if len(sys.argv)!=3:
    raise Exception('Error: plt_hyper_srch.py requires 2 arguments: model_stem max_n_svs')

# Import Parameters from json file
model_stem=sys.argv[1]
max_n_svs=int(sys.argv[2]) # the max number of support vectors

path_dict=ief.get_path_dict()
print(path_dict.keys())

#model_stem='genSvmSe'
#model_stem='genSvmEqWtsAes'
####model_stem='svmKdsampSeAes'
#model_stem='tempEqWt'
#model_stem='genLogregSe'
#model_dir='/home/dgroppe/GIT/SZR_ANT/MODELS/'
model_dir=os.path.join(path_dict['szr_ant_root'],'MODELS')
#model_dir='/Users/davidgroppe/PycharmProjects/SZR_ANT/MODELS/'
C_srch=list()
valid_bal_acc_srch=list()
train_bal_acc_srch=list()
nsvec_srch=list()
model_names_srch=list()
gamma_srch=list()
for f in os.listdir(model_dir):
    f_splt=f.split('_')
    if f_splt[0]==model_stem:
        # model of desired type
        in_fname=os.path.join(model_dir,f,'classify_metrics_srch.npz')
        if os.path.isfile(in_fname):
            temp=np.load(in_fname)
            C_srch=C_srch+list(temp['tried_C'])
    #         C_srch=C_srch+list(temp['C_srch'])
    #         temp_gamma=[temp['gamma'] for a in range(len(temp['C_srch']))]
            gamma_srch=gamma_srch+list(temp['tried_gamma'])
            valid_bal_acc_srch=valid_bal_acc_srch+list(temp['tried_valid_acc'])
            train_bal_acc_srch=train_bal_acc_srch+list(temp['tried_train_acc'])
            if 'tried_train_nsvec' in temp.keys():
                nsvec_srch=nsvec_srch+list(temp['tried_train_nsvec'])
            else:
                nsvec_srch = nsvec_srch + list(np.zeros(len(temp['tried_valid_acc'])))
            model_names_srch=model_names_srch+[f for a in range(len(temp['tried_C']))]

mx_id=np.argmax(valid_bal_acc_srch)
print('%d hyperparameters tried' % len(valid_bal_acc_srch))
print('Best Validation Accuracy: %f' % valid_bal_acc_srch[mx_id])
print('Best Training Accuracy: %f' % train_bal_acc_srch[mx_id])
print('Using C=%f, gam=%f, nsvec=%d' % (C_srch[mx_id],gamma_srch[mx_id],nsvec_srch[mx_id]))
print('Best model name is %s' % model_names_srch[mx_id])
print()

# Find best performance given limited # of support vectors
# Number of support vectors allowed:  200 support vectors with 125 dimensions
# Right now I have 30 dimensions so I can have 295 support vectors
#C_bool=np.asarray(C_srch)>5**10
nsvec_bool=np.asarray(nsvec_srch)>max_n_svs # 295 is max # of support vectors with 30d features
valid_bacc_lim=np.copy(valid_bal_acc_srch)
valid_bacc_lim[nsvec_bool]=0
lim_mx_id=np.argmax(valid_bacc_lim)
# print('Best Validation Accuracy C<5**10: %f' % valid_bacc_lim[lim_mx_id])
# print('Best Training Accuracy: %f' % train_bal_acc_srch[lim_mx_id])
# print('Using C=%f, gam=%f (C<5**10)' % (C_srch[lim_mx_id],gamma_srch[lim_mx_id]))
# print('Best model name is %s (C<5**10)' % model_names_srch[lim_mx_id])
print('Best Validation Accuracy (limited # of svecs): %f' % valid_bacc_lim[lim_mx_id])
print('Best Training Accuracy: %f (limited # of svecs)' % train_bal_acc_srch[lim_mx_id])
print('Using C=%f, gam=%f, nsvec=%d' % (C_srch[lim_mx_id],gamma_srch[lim_mx_id],nsvec_srch[lim_mx_id]))
print('Best model name is %s (limited # of svecs)' % model_names_srch[lim_mx_id])

print('Black circle in plot represents best accuracy (ignoring # of SVs)')
print('Green star in plot represents best accuracy with realistic # of SVs')

min_nsv_id=np.argmin(nsvec_srch)
print()
print('Smallest # of svecs: %f' % nsvec_srch[min_nsv_id])
print('Validation Accuracy: %f' % valid_bacc_lim[min_nsv_id])
print('Training Accuracy: %f' % train_bal_acc_srch[min_nsv_id])
print('Using C=%f, gam=%f' % (C_srch[min_nsv_id],gamma_srch[min_nsv_id]))
print('Corresponding model name is %s' % model_names_srch[min_nsv_id])


# Plot # of support vectors as a Function of Hyperparameters
if len(nsvec_srch)>0:
    plt.figure(2)
    plt.clf()
    #plt.scatter(np.arange(len(vbacc_ray)),vbacc_ray,c=vbacc_ray)
    plt.scatter(np.log10(C_srch),np.log10(gamma_srch),c=nsvec_srch,cmap='plasma',s=32)
    plt.plot(np.log10(C_srch[mx_id]),np.log10(gamma_srch[mx_id]),'k.')
    plt.plot(np.log10(C_srch[lim_mx_id]),np.log10(gamma_srch[lim_mx_id]),'g*')
    plt.xlabel('log10(C)')
    plt.ylabel('log10(Gamma)')
    cbar=plt.colorbar()
    cbar.set_label('# of Support Vectors', rotation=90)
    plt.show()

# Plot Accuracy as a Function of Hyperparameters
plt.figure(1)
plt.clf()
plt.subplot(1,3,1)
#plt.scatter(np.arange(len(vbacc_ray)),vbacc_ray,c=vbacc_ray)
plt.scatter(np.log10(C_srch),np.log10(gamma_srch),c=valid_bal_acc_srch,cmap='plasma',s=32)
plt.plot(np.log10(C_srch[mx_id]),np.log10(gamma_srch[mx_id]),'k.')
plt.plot(np.log10(C_srch[lim_mx_id]),np.log10(gamma_srch[lim_mx_id]),'g*')
plt.xlabel('log10(C)')
plt.ylabel('log10(Gamma)')
cbar=plt.colorbar()
#cbar.set_label('Balanced Accuracy', rotation=90)
plt.title('Validation Data')
#plt.show()

# plt.figure(2)
# plt.clf()
plt.subplot(1,3,2)
#plt.scatter(np.arange(len(vbacc_ray)),vbacc_ray,c=vbacc_ray)
plt.scatter(np.log10(C_srch),np.log10(gamma_srch),c=train_bal_acc_srch,cmap='plasma',s=32)
plt.plot(np.log10(C_srch[mx_id]),np.log10(gamma_srch[mx_id]),'k.')
plt.plot(np.log10(C_srch[lim_mx_id]),np.log10(gamma_srch[lim_mx_id]),'g*')
plt.xlabel('log10(C)')
# plt.ylabel('log10(Gamma)')
cbar=plt.colorbar()
#cbar.set_label('Balanced Accuracy', rotation=90)
plt.title('Training Data')

plt.subplot(1,3,3)
#plt.scatter(np.arange(len(vbacc_ray)),vbacc_ray,c=vbacc_ray)
df=np.asarray(train_bal_acc_srch)-np.asarray(valid_bal_acc_srch)
plt.scatter(np.log10(C_srch),np.log10(gamma_srch),c=df,cmap='plasma',s=32)
plt.plot(np.log10(C_srch[mx_id]),np.log10(gamma_srch[mx_id]),'k.')
plt.plot(np.log10(C_srch[lim_mx_id]),np.log10(gamma_srch[lim_mx_id]),'g*')
plt.xlabel('log10(C)')
# plt.ylabel('log10(Gamma)')
cbar=plt.colorbar()
cbar.set_label('Balanced Accuracy', rotation=90)
plt.title('Training-Validation')
plt.show()