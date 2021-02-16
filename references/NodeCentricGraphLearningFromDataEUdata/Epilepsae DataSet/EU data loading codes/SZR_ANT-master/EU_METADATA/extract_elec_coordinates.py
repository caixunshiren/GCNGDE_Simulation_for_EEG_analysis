# # Code for importing electrode names and coordinates from html file
# ## Note that there appear to be some coordinates missing for some electrodes
import pandas as pd
import os
import sys
import ieeg_funcs as ief


## Start of main function
if len(sys.argv)==1:
    print('Usage: extract_elec_coordinates.py sub_id')
    exit()
if len(sys.argv)!=2:
    raise Exception('Error: extract_elec_coordinates.py requires 1 argument: sub_id')

# Import Parameters from json file
sub=sys.argv[1]
print(sub)
print(type(sub))

path_dict=ief.get_path_dict()

#metadata_dir='/Users/davidgroppe/PycharmProjects/SZR_ANT/EU_METADATA/ELEC_HTML/' #get from path dict
metadata_dir=os.path.join(path_dict['szr_ant_root'],'EU_METADATA','ELEC_HTML')
html_fname=os.path.join(metadata_dir,'elecs_'+sub+'.html')
print('Attempting to read %s' % html_fname)
htable=pd.read_html(html_fname)

out_fname=os.path.join(path_dict['szr_ant_root'],'EU_METADATA','ELEC_COORD','elec_coord_'+sub+'.csv');

if os.path.isfile(out_fname):
# if out_fname.is_file():
    print('File exists: %s' % out_fname)
    print('Not over-writing as there may be manual edits.')
else:
    print('Creating file: %s' % out_fname)
    htable[0].to_csv(out_fname)






