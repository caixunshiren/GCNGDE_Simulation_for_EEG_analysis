function full_data_fname=get_data_fullfname(sub_id,hdr_fname, root)
%function full_data_fname=get_data_fullfname(sub_id,hdr_fname)
%
% Given the subject id and the name of a header file, this function returns
% the fullpath and filename of the corresponding data file on the external
% hard drive.
%
% Example Inputs:
% sub_id=1096;
% hdr_fname='109600102_0000.head';
%
% Example Outputs:
% full_data_fname =
%   /Volumes/ValianteLabEuData/EU/inv/pat_FR_1096/adm_1096102/rec_109600102/109600102_0072.data

dot_id=find(hdr_fname=='.');
data_fname=[hdr_fname(1:(dot_id-1)) '.data'];


%% Get directories where EU data might be stored. For some reason some
% patients have many of them
inv_subs=[1084, 1096, 1146, 253, 264, 273, 375, 384, 548, 565, 583, 590, 620, 862, 916, 922, 958, 970];
inv_subs2=[1073, 1077, 1125, 115, 1150, 139, 442, 635, 818];
inv_subs3=[13089, 13245, 732];
if ~isempty(intersect(sub_id,inv_subs)),
    inv_dir='inv';
elseif ~isempty(intersect(sub_id,inv_subs2)),
    inv_dir='inv2';
elseif ~isempty(intersect(sub_id,inv_subs3)),
    inv_dir='inv3';
else
    error('Could not find sub %d in inv, inv2, or inv3 subdirectories on external hard drive.', ...
        sub_id);
end

if ismac,
    ieeg_root_dir=fullfile(external_root,'SgateOSExJnld','EU_TEMP',inv_dir, ...
        sprintf('pat_FR_%d',sub_id),sprintf('adm_%d102',sub_id));
else
    ieeg_root_dir=fullfile(root,inv_dir, ...
        sprintf('pat_FR_%d',sub_id),sprintf('adm_%d102',sub_id));
end
ieeg_dirs=get_eu_data_dirs(ieeg_root_dir);
n_ieeg_dirs=length(ieeg_dirs);
%        /Volumes/ValianteLabEuData/EU/inv/pat_FR_1096/adm_1096102/rec_109600102/109600102_0072.data

file_dir=[];
for subdir_loop=1:n_ieeg_dirs,
    if exist(fullfile(ieeg_dirs{subdir_loop},data_fname),'file'),
        file_dir=ieeg_dirs{subdir_loop};
        break;
    end
end
if isempty(file_dir),
    error('Could not find %s in rec_* subpaths of %s',data_fname, ...
        ieeg_root_dir);
end

full_data_fname=fullfile(file_dir,data_fname);

