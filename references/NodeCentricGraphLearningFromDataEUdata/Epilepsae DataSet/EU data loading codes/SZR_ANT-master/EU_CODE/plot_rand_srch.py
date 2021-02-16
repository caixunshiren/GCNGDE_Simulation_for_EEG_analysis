import numpy as np
import sys
import pandas as pd
import os
import pickle
import scipy.io as sio
import ieeg_funcs as ief
# import re
import dgFuncs as dg
from sklearn import preprocessing
# from scipy import stats
# from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



if len(sys.argv)==1:
    print('Usage: plot_rand_srch.py model_stem (e.g., eu_svm_sbox')
    exit()
if len(sys.argv)!=2:
    raise Exception('Error: plot_rand_srch.py requires 1 argument: eu_svm_sbox')

# Import Parameters from json file
model_stem=sys.argv[1]
print('Importing model parameters from %s' % model_stem)

#/home/dgroppe/GIT/SZR_ANT/MODELS/eu_svm_sbox7_1096
in_path='/home/dgroppe/GIT/SZR_ANT/MODELS/'
model_names=list()
for fname in os.listdir(in_path):
    if fname.startswith(model_stem):
        model_names.append(fname)
    
print('Model names: {}'.format(model_names))
C_list=list()
g_list=list()
vbacc_list=list()
for mname in model_names:
    metrics_fname=os.path.join(in_path,mname,'classification_metrics.npz')
    metrics=np.load(metrics_fname)
    C_list=C_list+metrics['C_list'].tolist()
    g_list=g_list+metrics['g_list'].tolist()
    vbacc_list=vbacc_list+metrics['valid_bal_acc_list'].tolist()

# C_list=metrics['C_list']
C_ray=np.asarray(C_list)
# print(C_ray)
# g_list=metrics['g_list']
g_ray=np.asarray(g_list)
# print(g_ray)
# vbacc_list=metrics['valid_bal_acc_list']
vbacc_ray=np.asarray(vbacc_list)
# print(vbacc_ray)

# plt.figure(1)
# plt.plot(vbacc_ray,'-o')

print('Max validation balanced accuracy is %f' % np.max(vbacc_ray))
argmx=np.argmax(vbacc_ray)
print('Using C=%f and gamma=%f' % (C_ray[argmx],g_ray[argmx]) )
print('i.e., log10(C)=%f and log10(gamma)=%f' % (np.log10(C_ray[argmx]),np.log10(g_ray[argmx])) )

# fig=plt.figure(2)
# plt.clf()
# ax = fig.add_subplot(111, projection='3d')
# plt.plot(np.log10(C_ray),np.log10(g_ray),vbacc_ray,'.')
# ax.set_xlabel('log10(C)')
# ax.set_ylabel('log10(Gamma)')
# plt.show()
fig=plt.figure(3)
plt.clf()
plt.scatter(np.log10(C_ray),np.log10(g_ray),c=vbacc_ray,s=32)
plt.xlabel('log10(C)')
plt.ylabel('log10(Gamma)')
plt.title('Smart Random Search Validation Performance: '+model_stem)
cbar=plt.colorbar()
cbar.set_label('Balanced Accuracy', rotation=90)
plt.show()


