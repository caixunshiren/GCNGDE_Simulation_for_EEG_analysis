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
# str(perf['try_C'])
# xlabs=list()
# for c in perf['try_C']:
#     xlabs.append(str(c))
# print('C values tried: {}'.format(xlabs))
C_vals=perf['C_vals']
C_sort_ids=np.argsort(C_vals)

# Valid Sensitivity and Specificity
plt.figure(1)
plt.clf()
dat=perf['valid_sens']
_=plt.errorbar(C_vals[C_sort_ids],np.mean(dat[:,C_sort_ids],axis=0),np.std(dat[:,C_sort_ids],axis=0),label='Sensitivity')
dat=perf['valid_spec']
_=plt.errorbar(C_vals[C_sort_ids],np.mean(dat[:,C_sort_ids],axis=0),np.std(dat[:,C_sort_ids],axis=0),label='Specificity')
dat=perf['pptn_missed_szrs']
plt.xlabel('C')
plt.xscale('log')
plt.ylim([-.03, 1.03])
# plt.xticks(np.arange(len(perf['try_C'])),xlabs)
plt.title('Validation Data')
plt.legend(loc='best')
plt.show()


# Balanced Accuracy
plt.figure(2)
plt.clf()
dat=perf['train_bal_acc']
_=plt.errorbar(C_vals[C_sort_ids],np.mean(dat[:,C_sort_ids],axis=0),np.std(dat[:,C_sort_ids],axis=0),label='Train')
dat=perf['valid_bal_acc']
_=plt.errorbar(C_vals[C_sort_ids],np.mean(dat[:,C_sort_ids],axis=0),np.std(dat[:,C_sort_ids],axis=0),label='Valid')
xlim=plt.xlim()
_=plt.plot(xlim,[0.5, 0.5],'k--')
plt.ylabel('Balanced Accuracy')
plt.ylim([-.03, 1.03])
plt.xlabel('C')
# plt.xticks(np.arange(len(perf['try_C'])),xlabs)
plt.xscale('log')
plt.legend(loc='best')
plt.show()
