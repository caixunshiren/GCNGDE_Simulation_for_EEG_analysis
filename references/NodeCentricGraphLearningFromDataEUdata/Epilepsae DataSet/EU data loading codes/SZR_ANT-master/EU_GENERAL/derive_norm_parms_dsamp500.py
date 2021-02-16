import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ieeg_funcs as ief


# Get file list & figure out max and min feature values
in_fnames=list() # List of kdownsampled*.npz files (1 per sub)
sub_list=list() # Corresponding list of subs
# find all subsamp files
path_dict=ief.get_path_dict()
in_dir=os.path.join(path_dict['szr_ant_root'],'EU_GENERAL','KDOWNSAMP')
#in_dir='/Users/davidgroppe/PycharmProjects/SZR_ANT/EU_GENERAL/KDOWNSAMP/'
f_ct=0
for f in os.listdir(in_dir):
    if f.startswith('kdownsampled_'):
        in_fnames.append(f)
        tmp=f.split('_')[1]
        sub_list.append(int(tmp.split('.')[0]))
        mat=np.load(os.path.join(in_dir,f))
        if f_ct==0:
            min_ftrs=np.min(mat['ftrs_dsamp'],axis=0) # 30 dim vector
            max_ftrs=np.max(mat['ftrs_dsamp'],axis=0) # 30 dim vector
        else:
            min_ftrs=np.minimum(min_ftrs,np.min(mat['ftrs_dsamp'],axis=0)) # 30 dim vector
            max_ftrs=np.maximum(max_ftrs,np.max(mat['ftrs_dsamp'],axis=0)) # 30 dim vector
        f_ct+=1
        
n_band=6
band_labels=['Delta','Theta','Alpha','Beta','Gamma','HGamma']
n_edm=5
n_ftr=n_band*n_edm
n_file=len(in_fnames)
print('%d files total' % n_file)

bin_edge_list=list()
n_bin_list=list()
n_targ_bin=500 #?? make bigger
ictal_p_list=list()
nonictal_p_list=list()
label_list=list()
bin_cntr_list=list()
acc_list=list()
for a in range(n_ftr):
    bin_cntr_list.append([])
    
# Create the histogram edges. Edges may differ across feature but are the same across subs
for edm_ct in range(n_edm):
    for b_ct in range(n_band):
        bin_edge_list.append(np.linspace(min_ftrs[b_ct+edm_ct*n_band],max_ftrs[b_ct+edm_ct*n_band],n_targ_bin))
        #bin_edge_list.append(np.logspace(min_ftrs[b_ct+edm_ct*n_band],max_ftrs[b_ct+edm_ct*n_band],n_targ_bin))
        n_bin_list.append(len(bin_edge_list[-1]))
        label_list.append(band_labels[b_ct]+'-EDM'+str(edm_ct))
        acc_list.append(np.zeros((n_file,n_bin_list[-1])))

        
# Loop over features and find best threshold for class discrimination
print('Computing accuracy for each feature using a range of thresholds...')
n_wind=np.zeros(n_ftr)
for f_ct, f in enumerate(in_fnames):
    print('Working on ftr %d/%d' % (f_ct+1,n_ftr))
    mat=np.load(os.path.join(in_dir,f))
    n_wind[f_ct]=mat['ftrs_dsamp'].shape[0]
    
    # Loop over features
    for edm_ct in range(n_edm):
    #for edm_ct in range(1):
        raw_ftrs=mat['ftrs_dsamp'].T
        for b_ct in range(n_band):
            sens=np.zeros(len(bin_edge_list[b_ct+edm_ct*n_band]))
            spec=np.zeros(len(bin_edge_list[b_ct+edm_ct*n_band]))
            for thresh_ct, thresh in enumerate(bin_edge_list[b_ct+edm_ct*n_band]):
                    y_hat=(raw_ftrs[b_ct+edm_ct*n_band,:]>=thresh)
                    sens[thresh_ct]=np.mean(y_hat[mat['szr_class_dsamp']==1])
                    spec[thresh_ct]=np.mean(y_hat[mat['szr_class_dsamp']==0]==0)
            acc=np.abs(-.5+(sens+spec)/2)
            acc_list[b_ct+edm_ct*n_band][f_ct,:]=acc


print('Done!')

# Import normalization factors
#in_fname='/Users/davidgroppe/PycharmProjects/SZR_ANT/EU_GENERAL/KDOWNSAMP/ftr_limits.csv'
in_fname=os.path.join(path_dict['szr_ant_root'],'EU_GENERAL','KDOWNSAMP','ftr_limits.csv')
print('Loading normalization parameters from %s' % in_fname)
ftr_lims=pd.read_csv(in_fname,sep=',')
#ftr_lims.head()

# Loop over each feature and figure out what translation and division factor should be
cntr=np.zeros(n_ftr)
div_fact=np.zeros(n_ftr)
plt.figure(10)
plt.clf()
for f_ct in range(n_ftr):
    low_bnd=ftr_lims.iloc[f_ct,2]
    up_bnd=ftr_lims.iloc[f_ct,1]

    # Re-center bins
    ftr_vals=bin_edge_list[f_ct]
    ftr_vals[ftr_vals>up_bnd]=up_bnd
    ftr_vals[ftr_vals<low_bnd]=low_bnd
    rng=up_bnd-low_bnd;
    div_fact[f_ct]=rng/(2*3.99)
    ftr_vals=ftr_vals/div_fact[f_ct]
    
    cntr[f_ct]=-3.99-ftr_vals[0] # add this to feature value to make min possible value=-3.99
    ftr_vals=ftr_vals+cntr[f_ct]

    plt.plot(ftr_vals,np.mean(acc_list[f_ct],axis=0))
    plt.xlabel('Single Feature Threshold')
    plt.ylabel('Accuracy')
    plt.title('Mean Normalized Feature Accuracy')


plt.show()
# print(ftr_vals[0])
# print(ftr_vals[-1])    

# Save normalization parameters to disk
#out_fname='/Users/davidgroppe/PycharmProjects/SZR_ANT/EU_GENERAL/KDOWNSAMP/norm_factors.npz'
out_fname=os.path.join(path_dict['szr_ant_root'],'EU_GENERAL','KDOWNSAMP','norm_factors.npz')
print('Saving normalization parameters to %s' % out_fname)
np.savez(out_fname,cntr=cntr,div_fact=div_fact,ftr_labels=label_list,in_fnames=in_fnames)


