import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
import os
import ieeg_funcs as ief
import dgFuncs as dg
from sklearn import svm
from sklearn.externals import joblib
import sys

if len(sys.argv)==1:
    print('Usage: perf_smry_grid_srch.py model_name')
    exit()
if len(sys.argv)!=2:
    raise Exception('Error: train_srchC.py requires 1 argument: model_name')

model_name=sys.argv[1]
print('Model name is %s' % model_name)

path_dict=ief.get_path_dict()
print(path_dict.keys())
print(path_dict['szr_ant_root'])
in_file=os.path.join(path_dict['szr_ant_root'],'MODELS',model_name,'classify_metrics_srch.npz')
perf=np.load(in_file)

# Get xtick labels
str(perf['try_C'])
xlabs=list()
for c in perf['try_C']:
    xlabs.append(str(c))
print('C values tried: {}'.format(xlabs))

# Valid Sensitivity and Specificity
dat=perf['valid_sens']
_=plt.errorbar(np.arange(dat.shape[1]),np.mean(dat,axis=0),np.std(dat,axis=0),label='Sensitivity')
dat=perf['valid_spec']
_=plt.errorbar(np.arange(dat.shape[1]),np.mean(dat,axis=0),np.std(dat,axis=0),label='Specificity')
dat=perf['pptn_missed_szrs']
_=plt.errorbar(np.arange(dat.shape[1]),np.mean(dat,axis=0),np.std(dat,axis=0),label='Szrs Missed')
# _=plt.plot(np.arange(dat.shape[1]),np.mean(dat,axis=0),'r-o') #makes line visible is no
plt.xlabel('C')
plt.ylim([-.03, 1.03])
plt.xticks(np.arange(len(perf['try_C'])),xlabs)
plt.title('Validation Data')
plt.legend(loc='best')
plt.show()

# Balanced Accuracy
dat=perf['train_bal_acc']
_=plt.errorbar(np.arange(dat.shape[1]),np.mean(dat,axis=0),np.std(dat,axis=0),label='Train')
dat=perf['valid_bal_acc']
_=plt.errorbar(np.arange(dat.shape[1]),np.mean(dat,axis=0),np.std(dat,axis=0),label='Valid')
dat=perf['pptn_missed_szrs']
_=plt.errorbar(np.arange(dat.shape[1]),np.mean(dat,axis=0),np.std(dat,axis=0),label='Szrs Missed')
_=plt.plot([0, dat.shape[1]-1],[0.5, 0.5],'k--')
plt.ylabel('Balanced Accuracy')
plt.ylim([-.03, 1.03])
plt.xlabel('C')
plt.xticks(np.arange(len(perf['try_C'])),xlabs)
plt.legend(loc='best')
plt.show()


dat=perf['mn_stim_latency']
_=plt.errorbar(np.arange(dat.shape[1]),np.nanmean(dat,axis=0),np.std(dat,axis=0))
n_subs=dat.shape[0]
for a in range(dat.shape[1]):
    _=plt.plot(np.ones(n_subs)*a,dat[:,a],'o')
plt.xticks(np.arange(len(perf['try_C'])),xlabs)
plt.ylabel('Mean Stimulation Latency (0=~4 sec before clinician onset)')
plt.xlabel('C')
plt.show()

