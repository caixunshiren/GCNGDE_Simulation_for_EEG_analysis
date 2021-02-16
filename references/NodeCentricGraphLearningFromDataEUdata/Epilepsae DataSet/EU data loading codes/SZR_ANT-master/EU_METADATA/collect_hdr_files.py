# # Code for copying all EU iEEG header files into EU_METADATA subdirectory

import pandas as pd
import os
import sys
from pathlib import Path
from shutil import copyfile


# TODO read sub from cmnd line
## Start of main function
if len(sys.argv)==1:
    print('Usage: collect_hdr_files.py sub_number')
    exit()
if len(sys.argv)!=2:
    raise Exception('Error: collect_hdr_files.py requires 1 argument: sub_number')

# Import Parameters from json file
sub=int(sys.argv[1])

#TODO make this path machine-sensitive
eu_root='/media/dgroppe/ValianteLabEuData/EU/'
# Collect possible subject directories
inv_dirs=list()
for a in range(1,4):
    if a==1:
        inv_dirs.append(os.path.join(eu_root,'inv'))
    else:
        inv_dirs.append(os.path.join(eu_root,'inv'+str(a)))
print(inv_dirs)



# Find which inv* directory the subject is in
use_dir=[]
for sub_dir in inv_dirs:
    full_dir=Path(os.path.join(sub_dir,'pat_FR_'+str(sub)))
    if full_dir.is_dir():
        use_dir=full_dir
if use_dir==[]:
    print('Error inv* subdirectory not found for %d' % sub)
else:
    print(use_dir)




# Find all data subdirectories for this sub
# get adm subdir
for f in os.listdir(use_dir):
    if f.startswith('adm_'):
        adm_dir=os.path.join(use_dir,f)

# get all rec subdirs
rec_dirs=list()
for f in os.listdir(adm_dir):
    if f.startswith('rec_'):
        rec_dirs.append(os.path.join(adm_dir,f))
        


# Destination directory
#TODO make this machine independent
dest_dir=os.path.join('/home/dgroppe/GIT/SZR_ANT/EU_METADATA/IEEG_HEADERS/',str(sub)+'_headers')
if Path(dest_dir).is_dir():
    print('Warning: %s exists' % dest_dir)
else:
    os.mkdir(dest_dir)

# Loop over rec_dirs and cp over and header files 
for d in rec_dirs:
    cmnd='cp ' + os.path.join(d,'*.head') + ' ' + dest_dir
    os.system(cmnd)
    print(cmnd)
    #copyfile(os.path.join(d,'*.head'), dest_dir)


# metadata_dir='/Users/davidgroppe/PycharmProjects/SZR_ANT/EU_METADATA/ELEC_HTML/'
# html_fname=os.path.join(metadata_dir,'elecs_'+sub+'.html')
# print('Attempting to read %s' % html_fname)
# htable=pd.read_html(html_fname)



