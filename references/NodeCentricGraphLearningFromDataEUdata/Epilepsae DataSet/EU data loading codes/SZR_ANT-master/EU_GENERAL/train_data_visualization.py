""" This script loads the training features from a kdownsampled npz file and plots them in 3D using either PCA or T-SNE
to reduce the dimensionality. Ictal samples are in red. Interictal in blue. Each unique patient is represented with a unique
marker."""

import numpy as np
import scipy.io as sio
import os
import euGenFuncs as eu
import numpy as np
import pandas as pd
import pickle
import ieeg_funcs as ief
import dgFuncs as dg
import matplotlib.pyplot as plt
import imp 
from mpl_toolkits.mplot3d import Axes3D


infname='/home/dgroppe/GIT/SZR_ANT/EU_GENERAL/train_ftrs_unbnded_se_dsamp500.npz'
se=np.load(infname)
print(se['ftrs'].shape)
print(se.keys())

uni_sub=np.unique(se['sub_id'])
n_sub=len(uni_sub)
print('# of patients %d' % n_sub)
#print(uni_sub)

#show_tpts=np.arange(0,1000)
show_tpts=np.arange(0,se['ftrs'].shape[0],100)
show_szr_class=se['szr_class'][show_tpts]
show_sub_id=se['sub_id'][show_tpts]
print('Showing %d tpts' % len(show_tpts))

use_pca=False
if use_pca==True:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    dim3 = pca.fit_transform(se['ftrs'][show_tpts,:])
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    print('Total explained variance: {}'.format(np.sum(pca.explained_variance_ratio_)))
    xlab='PC1'
    ylab='PC2'
    zlab='PC3'
else:
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
    dim3 = tsne.fit_transform(se['ftrs'][show_tpts,:])
    print('TSNE done!')
    xlab='TSNE1'
    ylab='TSNE2'
    zlab='TSNE3'

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
rgb=('b','r')
mark=('s','o','^','v','8','+','*','h','x','D','<','>')
for ictal in range(2):
    tt=np.where(show_szr_class==ictal)
    for sub in range(n_sub):
        t=np.intersect1d(tt,np.where(show_sub_id==uni_sub[sub]))
        ax.scatter(dim3[t,0], dim3[t,1], dim3[t,2],c=rgb[ictal],marker=mark[sub])

ax.set_xlabel(xlab)
ax.set_ylabel(ylab)
ax.set_zlabel(zlab)

plt.show()

print('done!')