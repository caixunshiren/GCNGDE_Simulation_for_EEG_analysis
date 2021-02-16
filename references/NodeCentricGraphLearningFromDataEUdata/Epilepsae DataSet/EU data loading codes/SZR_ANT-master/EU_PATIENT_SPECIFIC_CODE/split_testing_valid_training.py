import numpy as np
import pandas as pd
import os
import pickle
import sys

if len(sys.argv)==1:
    print('Usage: split_testing_valid_training.py patient_id')
    exit()
if len(sys.argv)!=2:
    raise Exception('Error: split_testing_valid_training.py requires 1 argument: patient_id')

sub=sys.argv[1]
print('Patient being processed is %s' % sub)


# Import list of seizure times
# First 3 clinical srs, are training, next 2 clinical szrs validation, everything else is testing
df_fname='/Users/davidgroppe/Dropbox/TWH_INFO/EU_METADATA/szr_on_off_FR_'+sub+'.pkl'
szr_times_df=pickle.load(open(df_fname,'rb'))
szr_times_df.tail()


# Get index of clinical szrs in the data frame
clin_ids=szr_times_df[szr_times_df['SzrType']=='Clinical'].index.tolist()

# Get border between training-validation data and validation-testing data
# Training data contain the first 3 clinical szrs
# Validation data contain the next 2 clinical szrs
# Testing data consists of the rest
train_szr_offset=szr_times_df.loc[2].SzrOffsetSec
valid_szr_offset=szr_times_df.loc[3].SzrOffsetSec
valid_szr_onset=szr_times_df.loc[3].SzrOnsetSec
test_szr_onset=szr_times_df.loc[5].SzrOnsetSec

train_valid_border=int(np.round( (train_szr_offset+valid_szr_onset)/2 ))
print('Train-Valid border %d sec' % train_valid_border)
valid_test_border=int(np.round( (valid_szr_offset+test_szr_onset)/2 ))
print('Valid-Test border %d sec' % valid_test_border)


# Import mat file files to figure out which ones should be used in which data split
# Also keep track of which ones contain seizures (clinical or subclinical)
csv_fname='/Users/davidgroppe/Dropbox/TWH_INFO/EU_METADATA/data_on_off_FR_'+sub+'.csv'
mat_df=pd.read_csv(csv_fname)
mat_df=mat_df.drop('Unnamed: 0',1)


# Create list of file names for each data-split
train_files=list()
train_szr_files=list()
valid_files=list()
# Create list of file names containing szrs in each data-split
valid_szr_files=list()
test_files=list()
test_szr_files=list()
total_szrs=0
for row_id in range(mat_df.shape[0]):
    # See if the file contains szrs
    # Szrs with onsets after file onset
    post_ids=szr_times_df[szr_times_df['SzrOnsetSec']>=mat_df.iloc[row_id,2]].index.tolist()
    # Szrs with onsets before file offset
    pre_ids=szr_times_df[szr_times_df['SzrOnsetSec']<=mat_df.iloc[row_id,4]].index.tolist()
    n_szrs_in_file=len(np.intersect1d(post_ids,pre_ids))
    total_szrs+=n_szrs_in_file
    
    if mat_df.iloc[row_id,2]>test_szr_onset:
        # If file onset is after the valid-test border, make it test data
        test_files.append(mat_df.iloc[row_id,1].split('.')[0])
        if n_szrs_in_file>0:
            test_szr_files.append(test_files[-1])
    elif mat_df.iloc[row_id,2]>valid_szr_onset:
        # If file onset is after the train-valid border, make it validation data
        valid_files.append(mat_df.iloc[row_id,1].split('.')[0])
        if n_szrs_in_file>0:
            valid_szr_files.append(valid_files[-1])
    else:
        # Otherwise, make it training data
        train_files.append(mat_df.iloc[row_id,1].split('.')[0])
        if n_szrs_in_file>0:
            train_szr_files.append(train_files[-1])

print('%d training files' % (len(train_files)))
print('%d validation files' % (len(valid_files)))
print('%d testing files' % (len(test_files)))
print('%d total szrs found' % total_szrs)
print('There should be %d total szrs' % szr_times_df.shape[0])

# print(train_szr_files)
# print(valid_szr_files)
# print(test_szr_files)

# Save lists to pkl via a dict
out_fname='/Users/davidgroppe/Dropbox/TWH_INFO/EU_METADATA/data_splits_FR_'+sub+'.pkl'
split_dict={'train_szr_files': train_szr_files,'test_szr_files': test_szr_files,
            'valid_szr_files': valid_szr_files,'train_files': train_files,
            'test_files': test_files, 'valid_files': valid_files}
print('Saving lists of training, testing, & validation files to %s' % out_fname)
pickle.dump(split_dict,open(out_fname,'wb'))

