function subdirs=get_eu_data_dirs(root_dir)
%% Get directories where EU data might be stored. For some reason some 
% patients have many of them. They start with "rec_"
%
% root_dir is the root directory
% For example: /media/dgroppe/ValianteLabEuData/EU/inv/pat_FR_1096/adm_1096102/

f=dir(fullfile(root_dir,'rec_*'));
subdirs=cell(1,1);
subdir_ct=0;
for a=1:length(f),
    if isdir(fullfile(root_dir,f(a).name)),
        subdir_ct=subdir_ct+1;
        subdirs{subdir_ct}=fullfile(root_dir,f(a).name);
    end
end
fprintf('%d subdirectories of data found\n',subdir_ct);



