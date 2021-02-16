# This script runs apply_saved_models_to_szr.py on all subjects listed
# in test_subs.txt and train_subs.txt

import os
import sys


## Start of main function
if len(sys.argv)==1:
    print('Usage: batch_process.py model_name')
    exit()
if len(sys.argv)!=2:
    raise Exception('Error: batch_process.py takes one argument: model_name')

model_name=sys.argv[1]
print(model_name)
# exit()
# model_name='genLogregSe_1'

# Load test subjects
in_fname='test_subsAES.txt'
text_file=open(in_fname,'r')
temp=text_file.readlines()
subs=list()
for raw_sub in temp:
    subs.append(raw_sub.strip())

# Run all test subs
for sub in subs:
    cmnd='python apply_saved_models_to_szr.py '+sub+' ' + model_name
    os.system(cmnd)

# Load train subjects
in_fname='train_subs.txt'
text_file=open(in_fname,'r')
temp=text_file.readlines()
subs=list()
for raw_sub in temp:
    subs.append(raw_sub.strip())

# Run all train subs
for sub in subs:
    cmnd='python apply_saved_models_to_szr.py '+sub+' ' + model_name
    os.system(cmnd)