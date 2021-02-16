""" This script creates a csv file that indicates for each seed channel (row 1) the other 7 channels that
should be used for computing PLV. If only 8 channels are of interest, the problem is easy. If there are more than 8
channels, then the script selects the closest 7 to each seed channel."""

import pandas as pd
import os
import ieeg_funcs as ief
import pickle
import numpy as np
import sys

basePathSave = 'C:/Users/Nafiseh Ghoroghchian/Dropbox/PhD/EU data loading codes/SZR_ANT-master/'
## Start of main function
# if len(sys.argv)==1:
#     print('Usage: create_plv_chan_csv.py sub_id (e.g., 1077)')
#     exit()
# if len(sys.argv)!=2:
#     raise Exception('Error: create_plv_chan_csv.py requires 1 argument: sub_id')

# Import Parameters from json file
# sub_id=sys.argv[1]

# Load list of channels of interest (I limit this to 8 or the # of SOZ chans)
sub_id=442
in_fname= basePathSave + 'EU_METADATA/ANALYZE_CHANS/'+sub_id+'_bi_use_chans.txt'

coi_df=pd.read_csv(in_fname,header=None) #COI=Channels of Interest
coi_df.head()
n_coi=coi_df.shape[0]

out_fname=basePathSave +'EU_METADATA/PLV_CHANS/'+sub_id+'_plv.csv'
print('Creating file %s' % out_fname)
fid = open(out_fname,'w')

fid.write('Seed,')
for a in range(1,8):
    fid.write('Pair%d,' %a)
fid.write('\n')


print('# of channels of interest: %d' % n_coi)
# If 8, just make table that indicate that all other channels should be used as PLV channels
if n_coi==8:
    for a in range(n_coi):
        fid.write('%s,' % coi_df.iloc[a,0])
        ct=0
        for b in range(n_coi):
            if a!=b:
                ct+=1
                fid.write('%s' % coi_df.iloc[b,0])
                if ct<7:
                    fid.write(',')
        fid.write('\n')
    fid.close()
else:
    #If more than 8, find the closest 8 other channels and use those as plv channels

    # Import list of SOZ channels
    soz_fname = basePathSave +'EU_METADATA/SOZ_CHANS/'+sub_id+'_bi_soz_chans.txt'
    soz_df = pd.read_csv(soz_fname, header=None)
    soz_df.head()
    n_soz = soz_df.shape[0]
    print('# of soz chans %d' % n_soz)

    # Import electrode coordinates
    xyz_fname = basePathSave +'EU_METADATA/ELEC_COORD/elec_coord_'+sub_id+'.csv'
    xyz_df=pd.read_csv(xyz_fname)
    print(xyz_df.head())
    n_xyz=xyz_df.shape[0]

    # Check for missing coordinates and replace with 0
    for row_ct in range(n_xyz):
        if xyz_df.iloc[row_ct,4]=='-':
            print('Chan %s, Replacing missing coords with 0 0 0' % xyz_df.iloc[row_ct,0])
            for a in range(3):
                xyz_df.iloc[row_ct,4+a]='0'

    # Create coordinates for bipolar channels
    xyz_bi = np.zeros((n_soz, 3))
    for soz_ct in range(n_soz):
        bi_chan = soz_df.iloc[soz_ct][0]
        print(bi_chan)
        mono_chans = bi_chan.split('-')
        chan1_id = -1
        chan2_id = -1
        for chan_ct in range(n_xyz):
            if xyz_df.iloc[chan_ct, 0] == mono_chans[0]:
                chan1_id = chan_ct
            if xyz_df.iloc[chan_ct, 0] == mono_chans[1]:
                chan2_id = chan_ct
        if chan1_id < 0:
            print('Error: could not find coords for chan %s' % mono_chans[0])
        if chan2_id < 0:
            print('Error: could not find coords for chan %s' % mono_chans[1])
        chan_abs_sum = np.zeros(2)
        for dim_ct in range(3):
            chan_abs_sum[0] += np.abs(float(xyz_df.iloc[chan1_id, dim_ct + 4]))
            chan_abs_sum[1] += np.abs(float(xyz_df.iloc[chan2_id, dim_ct + 4]))
            xyz_bi[soz_ct, dim_ct] = (float(xyz_df.iloc[chan1_id, dim_ct + 4]) + float(
                xyz_df.iloc[chan2_id, dim_ct + 4])) / 2
        for a in range(2):
            if chan_abs_sum[a] == 0:
                print('WARNING: Coords for %s are bogus' % mono_chans[a])

    # For each SOZ channel find 7 closest other SOZ channels
    for seed_ct in range(n_soz):
        # For this channel measure distance will all other SOZ channels
        dst = np.zeros(n_soz)
        for soz_ct in range(n_soz):
            dif = xyz_bi[seed_ct, :] - xyz_bi[soz_ct, :]
            dst[soz_ct] = np.sqrt(np.sum(dif ** 2))

        # Sort by distance
        sort_ids = np.argsort(dst)

        # Output closest 8 (first one will be the channel with itself)
        fid.write('%s,' % soz_df.iloc[seed_ct][0])
        for a in range(1, 8):
            if a<7:
                fid.write('%s,' % soz_df.iloc[sort_ids[a]][0])
            else:
                fid.write('%s\n' % soz_df.iloc[sort_ids[a]][0])
    fid.close()

print('Done.')



