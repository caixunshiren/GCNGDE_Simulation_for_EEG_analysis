function cli_szr_info=get_szr_fnames(sub_id, git_root, external_root)
%function cli_szr_info=get_szr_fnames(sub_id)
% 
% Extracts information about each clinical szr for a particular patient.
%
% sub_id - subject id (e.g., 1096)
% file_dir - path to where the ieeg data are stored
%
% Output:
% cli_szr_info =  struct array with fields:
%     clinical_fname=path and filename of raw ieeg clip
%     clinical_onset_sec=onset of seizure in seconds relative to the start
%                        of the clip
%     clinical_offset_sec=offset of seizure in seconds relative to the end
%                         of the clip
%     clinical_szr_num=index of seizure in seizure csv file (e.g., ??)
%     clinical_soz_chans=struct array of SOZ channels


%% Load szr onset and offset times
szr_times_csv=fullfile(git_root,'EU_METADATA','SZR_TIMES',['szr_on_off_FR_' num2str(sub_id) '.csv']);
szr_times=csv2Cell(szr_times_csv,',',1);
n_szrs=size(szr_times,1);

clinical_ids=[];
for a=1:n_szrs,
    if strcmpi(szr_times{a,7},'Clinical'),
        clinical_ids=[clinical_ids a];
    end
end
n_clinical=length(clinical_ids);
fprintf('%d clinical szrs\n',n_clinical);

%% Get directories where EU data might be stored. For some reason some 
inv_dir = get_inv_dir (sub_id);

% internal_pat_ids =[253,264,273,375,384,548];
% external_pat_ids = [565,583,590,620,862,916,922,958,970,1084,1096,1146 ...
%                         115,1073,1077,1125,1150];

% if ismac,
    % ?? TODO Temp hack, remove!!!
    %ieeg_root_dir='/Volumes/SgateOSExJnld/EU_TEMP/pat_FR_1096/adm_1096102';
    %ieeg_root_dir='/Volumes/SgateOSExJnld/EU_TEMP/inv/pat_FR_1096/adm_1096102';
%     ieeg_root_dir='/Volumes/ValianteLabEuData/EU/inv/pat_FR_1096/adm_1096102';
% end

ieeg_root_dir=fullfile(external_root,inv_dir, ...
        sprintf('pat_FR_%d',sub_id),sprintf('adm_%d102',sub_id));
ieeg_dirs=get_eu_data_dirs(ieeg_root_dir);
n_ieeg_dirs=length(ieeg_dirs);
% '/Volumes/ValianteLabEuData/EU/inv/pat_FR_620/adm_620102';
% '/media/dgroppe/ValianteLabEuData/EU/inv/pat_FR_1096/adm_1096102/

%% Get onset and offset times of clinical seizures
clinical_onset_sec=zeros(n_clinical,1);
clinical_offset_sec=zeros(n_clinical,1);
clinical_onset_sec=zeros(n_clinical,1);
clinical_offset_sec=zeros(n_clinical,1);
clinical_szr_num=zeros(n_clinical,1);
clinical_soz_chans=cell(n_clinical,1);
for a=1:n_clinical,
    temp_str=szr_times{clinical_ids(a),2};
    % Remove nuisance characters
    temp_str=strrep(temp_str,'''','');
    temp_str=strrep(temp_str,'[','');
    temp_str=strrep(temp_str,']','');
    clinical_soz_chans{a}=strsplit(temp_str,' ');
    clinical_szr_num(a)=str2num(szr_times{clinical_ids(a),1});
    clinical_offset_sec(a)=str2num(szr_times{clinical_ids(a),3});
    clinical_onset_sec(a)=str2num(szr_times{clinical_ids(a),5});
end


%% Load list of file start and stop times
file_times_csv=fullfile(git_root,'EU_METADATA','IEEG_ON_OFF',['data_on_off_FR_' num2str(sub_id) '.csv']);
file_times=csv2Cell(file_times_csv,',',1);
n_files=size(file_times,1);
fprintf('%d data files\n',n_files);

% Get onset and offset times for each file
file_onset_sec=zeros(n_files,1);
file_offset_sec=zeros(n_files,1);
file_fname=cell(n_files,1);
file_duration_sec=zeros(n_files,1);
for a=1:n_files,
    file_onset_sec(a)=str2num(file_times{a,4});
    file_offset_sec(a)=str2num(file_times{a,6});
    file_fname{a}=file_times{a,3};
    file_duration_sec(a)=str2num(file_times{a,2});
end


%% For each clinical szr:
% -identify the file it is in
% -identify the onset/offset of the szr in the file 
% -identify the onset chans
clinical_fname=cell(1,n_clinical);
clear cli_szr_info
for a=1:n_clinical,
%for a=1:1,
    fprintf('Trying to import clinical szr#%d\n',a);
    % Find out which file the szr is in
    file_id=0;
    for file_pointer=1:n_files,
        if clinical_onset_sec(a)>=file_onset_sec(file_pointer) && ...
                clinical_onset_sec(a)<file_offset_sec(file_pointer)
            file_id=file_pointer;
            break;
        end
    end
    if file_id==0,
        fprintf('Could NOT find szr#%d!!!!!!!!!!!\n',a);
    else
        hdr_fname=file_fname{file_pointer};
        hdr_stem=hdr_fname(1:(find(hdr_fname=='.')-1));
        data_fname=[hdr_stem '.data'];
        fprintf('Szr is in file %s\n',data_fname);
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
        cli_szr_info(a).clinical_fname=fullfile(file_dir,data_fname);
        cli_szr_info(a).clinical_onset_sec=clinical_onset_sec(a)-file_onset_sec(file_id);
        cli_szr_info(a).clinical_offset_sec=clinical_offset_sec(a)-file_onset_sec(file_id);
        cli_szr_info(a).clinical_szr_num=clinical_szr_num(a);
        cli_szr_info(a).clinical_soz_chans=clinical_soz_chans{a};
        %         clinical_fname{a}=fullfile(file_dir,data_fname);
        %         clinical_onset_sec(a)=clinical_onset_sec(a)-file_onset_sec(file_id);
        %         clinical_offset_sec(a)=clinical_offset_sec(a)-file_onset_sec(file_id);
    end
end


