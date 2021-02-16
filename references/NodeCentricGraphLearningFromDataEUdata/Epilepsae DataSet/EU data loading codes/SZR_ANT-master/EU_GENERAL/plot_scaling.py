""" Plot accuracy as a function of threshold for each AES feature to determing most useful range of
values for range limited chip"""

import numpy as np
import scipy.io as sio
import os
import euGenFuncs as eu
import numpy as np
import pandas as pd
import pickle
import ieeg_funcs as ief
import dgFuncs as dg
from scipy.interpolate import UnivariateSpline
from scipy.stats import iqr
import matplotlib.pyplot as plt

in_file='/Users/davidgroppe/PycharmProjects/SZR_ANT/EU_GENERAL/KDOWNSAMP/kdsamp_scaling.npz'
npz=np.load(in_file)
print(npz.keys())

n_ftrs=len(npz['label_list'])

# Create a text file of the feature names and lowest data value that you can import into a spreadsheet to enter thresholds
fid=open('temp_ftr_labels.txt','w')
fid.write('FtrLabel, MinValue\n')
for a in range(n_ftrs):
    fid.write('%s, %f\n' % (npz['label_list'][a],npz['bin_edge_list'][a,0]))

# Plot the accuracy of each feature for discriminating ictal vs. nonictal time periods as a function of threshold
# This gives some sense of where the informative range of data values are
# Each plot is interactive. You need to close it before you get the next plot.
for a in range(n_ftrs):
    plt.figure(1)
    plt.clf()
    plt.plot(npz['bin_edge_list'][a], npz['acc_list'][a].T) # Data from each individual feature
    plt.plot(npz['bin_edge_list'][a], np.mean(npz['acc_list'][a], axis=0), 'k-') # mean of all subjects
    plt.title(npz['label_list'][a])
    plt.show()