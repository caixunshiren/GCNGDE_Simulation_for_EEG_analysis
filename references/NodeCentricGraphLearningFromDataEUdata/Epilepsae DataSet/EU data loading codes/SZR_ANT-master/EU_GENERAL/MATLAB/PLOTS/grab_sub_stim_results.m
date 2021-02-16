function stim_stats=grab_sub_stim_results(sub)
%function stim_stats=grab_sub_stim_results(sub)

path='/home/dgroppe/GIT/SZR_ANT/MODELS/seSvmDsamp500TestFullFinal_2';
stem=sprintf('%s/%s_edmwt*.mat',path,num2str(sub));
f=dir(stem);
n_files=length(f);
fprintf('%d total files found:\n',n_files);
if n_files==0
   error('Run compute_acc_with_refractoryh_edm.py dufus!'); 
end
for ff=1:n_files
   disp(f(ff).name); 
end
fprintf('\n\n');

clear stim_stats;
stim_stats.grand_fp_per_day=zeros(n_files,1);
stim_stats.grand_edm_wts=zeros(n_files,1);
stim_stats.grand_mn_stim_lat=zeros(n_files,1);
stim_stats.grand_md_stim_lat=zeros(n_files,1);
stim_stats.pcnt_within_5sec=zeros(n_files,1);
stim_stats.pcnt_within_10sec=zeros(n_files,1);
stim_stats.abs_stim_lat_sec=[];
%load('/home/dgroppe/GIT/SZR_ANT/MODELS/seSvmDsamp500TestFullFinal_2/264_edmwt_10_refract_30_stim_results.mat');

for floop=1:n_files
   load(fullfile(path,f(floop).name));
   stim_stats.grand_fp_per_day(floop)=fp_per_hour*24;
   stim_stats.grand_edm_wts(floop)=edm_wt;
   abs_stim_lat=abs(stim_lat);
   stim_stats.abs_stim_lat_sec{floop}=abs_stim_lat;
   stim_stats.grand_mn_stim_lat(floop)=mean(abs_stim_lat);
   stim_stats.grand_md_stim_lat(floop)=median(abs_stim_lat);
   stim_stats.pcnt_within_5sec(floop)=mean(abs_stim_lat<=5)*100;
   stim_stats.pcnt_within_10sec(floop)=mean(abs_stim_lat<=10)*100;
end

%% Sort by edm wt
[stim_stats.grand_edm_wts, sort_id]=sort(stim_stats.grand_edm_wts);
stim_stats.grand_fp_per_day=stim_stats.grand_fp_per_day(sort_id);
stim_stats.abs_stim_lat_sec=stim_stats.abs_stim_lat_sec(sort_id);
stim_stats.grand_mn_stim_lat=stim_stats.grand_mn_stim_lat(sort_id);
stim_stats.grand_md_stim_lat=stim_stats.grand_md_stim_lat(sort_id);
stim_stats.pcnt_within_5sec=stim_stats.pcnt_within_5sec(sort_id);
stim_stats.pcnt_within_10sec=stim_stats.pcnt_within_10sec(sort_id);
