# This script imports the start and stop times for all patient data files and converts them to seconds relative to Jan 1, 2000
# This makes it easier to relate szr onset/offset times to the data
# This script outputs a csv file with the following columns:
#'HeaderFname': name of the *.header from file from which times were taken
#'StartSec': file onset in seconds from Jan. 1, 2000
#'StopSec': end of file in seconds from Jan. 1, 2000
#'DurationSec': how many seconds long the file is. Most files are 1 hour long (3600), but some are shorter
#'StartStr': The timing of file onset as a string (taken directly from the header file)

import pandas as pd
import os
import sys
import numpy as np
from datetime import datetime
# import ieeg_funcs as ief

# if len(sys.argv)==1:
#     print('Usage: export_EU_data_times.py sub# (e.g., export_EU_data_times.py 1146')
#     exit()
# if len(sys.argv)!=2:
#     raise Exception('Error: export_EU_data_times.py requires 1 argument: the patient # (e.g., 1146)')


# sub=sys.argv[1]

# Get list of header files
# path_dict=ief.get_path_dict()
basePathSave = 'C:/Users/Nafiseh Ghoroghchian/Dropbox/PhD/EU data loading codes/SZR_ANT-master/'
sub=str(1073)
#header_dir='/Users/davidgroppe/Dropbox/TWH_INFO/EU_METADATA/'+sub+'_headers/'
#header_dir='/Users/davidgroppe/PycharmProjects/SZR_ANT/EU_METADATA/IEEG_HEADERS/'+sub+'_headers/'
header_dir=os.path.join(basePathSave,'EU_METADATA/IEEG_HEADERS/'+sub+'_headers/')
temp_files=os.listdir(header_dir)
hdr_files=list()
for a in temp_files:
    if a.endswith('.head'):
        hdr_files.append(a)
n_hdr=len(hdr_files)
print('%d header files found' % n_hdr)


start_sec=list()
stop_sec=list()
start_str=list()
stop_str=list()
file_dur=list()
file_gaps=np.zeros(n_hdr)

#milestone=datetime(1970,1,1)
milestone=datetime(2000,1,1) # Arbirary date via which to convert times into seconds
#milestone=datetime(2009,6,21)
gap_ct=0
for hdr_fname in hdr_files:
    in_fname=os.path.join(header_dir,hdr_fname)
    hdr_df=pd.read_csv(in_fname,sep='=',header=None,names=['Type','Value'])
    start_str.append(hdr_df.iloc[0,1])
    dat_splt=start_str[-1].split(' ')
    
    temp_dt=datetime.strptime(dat_splt[0],'%Y-%m-%d')
    ttl_sec=(temp_dt-milestone).total_seconds() # # of seconds since Jan. 1 2000 since the day started
    hr_splt=dat_splt[1].split(':')
    ttl_sec+=int(hr_splt[0])*3600+int(hr_splt[1])*60+float(hr_splt[2])
    start_sec.append(ttl_sec)
    n_samp=int(hdr_df.iloc[1,1]) # of time points in the file
    Fs=int(hdr_df.iloc[2,1]) #sampling rate
    stop_sec.append(ttl_sec+n_samp/Fs) # offset of file in seconds
    file_dur.append(stop_sec[-1]-start_sec[-1]) #duration of file in seconds
    
    if len(stop_sec)>1:
        file_gaps[gap_ct]=start_sec[-1]-stop_sec[-2]
        gap_ct+=1


uni_gaps, uni_ct=np.unique(file_gaps,return_counts=True)
for loopy in range(len(uni_gaps)):
    print('Gap=%f sec, occurences=%d' % (uni_gaps[loopy],uni_ct[loopy]))



# Plot start and stop of each file to look for really big gaps
# plt.figure(1)
# plt.clf()
# # plt.plot([start_sec[0], stop_sec[-1]],[0, 0],'r-')
# for hdr_ct in range(n_hdr):
#     plt.plot([start_sec[hdr_ct], stop_sec[hdr_ct]],[1, 1],'b-')
# plt.xlim([start_sec[0], stop_sec[-1]])


# Create datafram
file_times=pd.DataFrame({'HeaderFname': hdr_files,
                        'StartSec': start_sec,
                        'StopSec': stop_sec,
                        'DurationSec':file_dur,
                        'StartStr': start_str})
# file_times.head()


# Output to csv so MATLAB can read it
#out_path='/Users/davidgroppe/PycharmProjects/SZR_ANT/EU_METADATA/IEEG_ON_OFF/'
out_path=os.path.join(basePathSave,'EU_METADATA/IEEG_ON_OFF/')
out_fname=os.path.join(out_path,'data_on_off_FR_'+str(sub)+'.csv')
print('Saving file to:')
print(out_fname)
file_times.to_csv(out_fname)




