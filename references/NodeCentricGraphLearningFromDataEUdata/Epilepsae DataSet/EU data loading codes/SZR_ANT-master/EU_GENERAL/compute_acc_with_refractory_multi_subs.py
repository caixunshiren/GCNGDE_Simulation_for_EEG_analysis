# This script runs apply_saved_models_to_szr.py on all subjects listed
# in test_subs.txt and train_subs.txt

import os
import sys
import numpy as np

## Start of main function
if len(sys.argv)==1:
    print('Usage: compute_acc_with_refractory_multi_subs.py model_name sub_list')
    exit()
if len(sys.argv)!=3:
    raise Exception('Error: compute_acc_with_refractory_multi_subs.py takes two arguments: model_name sub_list')

model_name=sys.argv[1]
print(model_name)
sub_list_fname=sys.argv[2]
print('Reading subjects to process from %s' % sub_list_fname)
# model_name='genLogregSe_1'

# Load test subjects
text_file=open(sub_list_fname,'r')
temp=text_file.readlines()
subs=list()
for raw_sub in temp:
    subs.append(raw_sub.strip())

# Run all test subs
for sub in subs:
    for thresh in np.arange(0.2,0.7,0.1):
        #cmnd='python apply_saved_models_to_szr.py '+sub+' ' + model_name
        cmnd = 'python compute_acc_with_refractory.py ' + sub + ' ' + str(thresh)+' '+ model_name +' False'
        print('Running: %s' % cmnd)
        os.system(cmnd)
