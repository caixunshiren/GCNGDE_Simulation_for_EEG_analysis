import os
import sys
import pandas as pd

## Start of main function
if len(sys.argv)==1:
    print('Usage: kmeans_dsamp_batch.py sub_list')
    exit()
if len(sys.argv)!=2:
    raise Exception('Error: kmeans_dsamp_batch.pyrequires 1 argument: sub_list')

# Import list of subjects to use
sub_list_fname=sys.argv[1]
print('Importing subs to use from %s' % sub_list_fname)
# use_subs_df=pd.read_csv(os.path.join(path_dict['szr_ant_root'],'use_subs.txt'),header=None,na_filter=False)
use_subs_df = pd.read_csv(sub_list_fname, header=None, na_filter=False)
train_subs_list = []
for sub in use_subs_df.iloc[:, 0]:
    train_subs_list.append(sub)

print('Subs to process: {}'.format(train_subs_list))

for sub in train_subs_list:
    os.system("python kmeans_dsamp_per_sub.py "+str(sub))