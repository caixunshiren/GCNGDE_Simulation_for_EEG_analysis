function file_info=get_fnames_and_szr_times(sub_id)
%function file_info=get_fnames_and_szr_times(sub_id)
% 
% Extracts information about each szr (clinical & subclinical) for a particular patient.
%
% sub_id - subject id (e.g., 1096)
% file_dir-path to where the ieeg data are stored
%
% Output:
% file_info =  struct array with fields:
%               fname: '109600102_0007.head'
%      file_onset_sec: [165x1 double]<-onset of the file in seconds
%      relative to an arbitrary date
%     file_offset_sec: [165x1 double]<-offset of the file in seconds
%      relative to an arbitrary date
%      file_onset_hrs: '2009-06-18 18:54:10.000'
%        file_dur_sec: [165x1 double]
%      szr_onsets_sec: 1082<-onset of any seizures in seconds relative to
%      the start of the file
%     szr_offsets_sec: 1119


%% Load szr onset and offset times
%'/home/dgroppe/GIT/SZR_ANT/
if ismac,
    % LAP
    git_root='/Users/davidgroppe/PycharmProjects/SZR_ANT/';
    external_root='/Volumes/';
else
    % MARR
    git_root='/home/dgroppe/GIT/SZR_ANT/';
    external_root='/media/dgroppe/';
end
szr_times_csv=fullfile(git_root,'EU_METADATA','SZR_TIMES',['szr_on_off_FR_' num2str(sub_id) '.csv']);

% First scan the file to make sure there are no " indicating SOZ electrode
% lists that are so long they go over multiple lines
fid=fopen(szr_times_csv,'r');
while ~feof(fid)
    tline = fgetl(fid);
    if sum(tline=='"'),
       error('%s has " in it. SOZ electrodes probably run over multiple lines. Fix it.', ...
           szr_times_csv);
    end
end
fclose(fid);

szr_times=csv2Cell(szr_times_csv,',',1);
n_szrs=size(szr_times,1);
fprintf('%d clinical+subclinical szrs\n',n_szrs);


%% Convert szr times to numeric variables
szr_onsets=zeros(n_szrs,1);
szr_offsets=zeros(n_szrs,1);
for sloop=1:n_szrs,
    if isempty(szr_times{sloop,5}),
        szr_onsets(sloop)=NaN;
    else
        szr_onsets(sloop)=str2num(szr_times{sloop,5});
    end
    if isempty(szr_times{sloop,3}),
        szr_offsets(sloop)=NaN;
    else
        szr_offsets(sloop)=str2num(szr_times{sloop,3});
    end
end


%% Load list of file start and stop times
file_times_csv=fullfile(git_root,'EU_METADATA','IEEG_ON_OFF',['data_on_off_FR_' num2str(sub_id) '.csv']);
file_times=csv2Cell(file_times_csv,',',1);
n_files=size(file_times,1);
fprintf('%d data files\n',n_files);


%% Convert files times to numeric variables
f_onsets=zeros(n_files,1);
f_offsets=zeros(n_files,1);
file_dur=zeros(n_files,1);
for floop=1:n_files,
    f_onsets(floop)=str2num(file_times{floop,4});
    f_offsets(floop)=str2num(file_times{floop,6});
    file_dur(floop)=str2num(file_times{floop,2});
end

%% Loop over all files and figure out onset/offset of szrs in the file (if any)
%file_info(100).FirstName = 'George';
% georgeStruct = struct('FirstName','George','Height', ...
%     {195 189 190 194 193})

% Initialize struct array with header file names
file_info=struct('fname',file_times(:,3),'file_onset_sec',f_onsets, ...
    'file_offset_sec',f_offsets,'file_onset_hrs',file_times(:,5),'file_dur_sec', ...
    file_dur);
for floop=1:n_files,
    sonsets_this_file=[];
    soffsets_this_file=[];
    for sloop=1:n_szrs,
        if (szr_onsets(sloop)>=f_onsets(floop)) && (szr_onsets(sloop)<=f_offsets(floop))
            % Szr onset is in this file
            if (szr_offsets(sloop)>=f_onsets(floop)) && (szr_offsets(sloop)<=f_offsets(floop))
                % Szr offset is also in this file
                sonsets_this_file=[sonsets_this_file szr_onsets(sloop)-f_onsets(floop)];
                soffsets_this_file=[soffsets_this_file szr_offsets(sloop)-f_onsets(floop)];
            else
                warning('Szr onset is in this file but NOT the offset. I will call all time points after onset "szr".');
                sonsets_this_file=[sonsets_this_file szr_onsets(sloop)-f_onsets(floop)];
                soffsets_this_file=[soffsets_this_file f_offsets(floop)-f_onsets(floop)];
            end
        elseif (szr_offsets(sloop)>=f_onsets(floop)) && (szr_offsets(sloop)<=f_offsets(floop))
            % Szr offset is in this file but NOT the onset
            warning('Szr offset is in this file but NOT the onset. I will call all time points before onset "szr".');
            sonsets_this_file=[sonsets_this_file 0];
            soffsets_this_file=[soffsets_this_file szr_offsets(sloop)-f_onsets(floop)];
        end
    end
    file_info(floop).szr_onsets_sec=sonsets_this_file;
    file_info(floop).szr_offsets_sec=soffsets_this_file;
end

