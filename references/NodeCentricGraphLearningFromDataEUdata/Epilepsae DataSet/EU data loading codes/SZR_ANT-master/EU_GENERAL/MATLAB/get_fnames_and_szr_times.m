function file_info=get_fnames_and_szr_times(sub_id, git_root)
%function file_info=get_fnames_and_szr_times(sub_id)
% 
% Extracts information about each szr (clinical & subclinical) for a particular patient.
%
% This differs from get_fnames_and_szr_time in that it discriminates
% between clinical and subclinical szrs
%
% sub_id - subject id (e.g., 1096)
%
% Output:
% file_info =  struct array with fields:
%               fname: '109600102_0007.head'
%      file_onset_sec: [165x1 double]<-onset of the file in seconds
%                      relative to an arbitrary date
%     file_offset_sec: [165x1 double]<-offset of the file in seconds
%                      relative to an arbitrary date
%      file_onset_hrs: '2009-06-18 18:54:10.000'
%        file_dur_sec: [165x1 double]
%      clin_szr_onsets_sec: 0
%     clin_szr_offsets_sec: 3600
%          clin_szr_csv_id: 8
%       sub_szr_onsets_sec: []
%      sub_szr_offsets_sec: []
%           sub_szr_csv_id: [] <-index of szr in sub's csv file (e.g.,
%           szr_on_off_FR_862.csv)
%      szr_onsets_sec: 1082<-onset of any seizures in seconds relative to
%           the start of the file
%     szr_offsets_sec: 1119
%
% Notes: 
% -this function used to be called get_fnames_and_szr_times2.m
% -files in file_info are sorted according to time of
% occurrence

%% Load szr onset and offset times
%'/home/dgroppe/GIT/SZR_ANT/
szr_times_csv=fullfile(git_root,'EU_METADATA','SZR_TIMES',['szr_on_off_FR_' num2str(sub_id) '.csv']);
szr_times=csv2Cell(szr_times_csv,',',1);
n_szrs=size(szr_times,1);
fprintf('%d clinical+subclinical szrs\n',n_szrs);


%% Convert szr times to numeric variables, NaN if not given
szr_onsets=zeros(n_szrs,1);
szr_offsets=zeros(n_szrs,1);
clin_szr=zeros(n_szrs,1);
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
    if strcmpi(szr_times{sloop,7},'Clinical'),
        clin_szr(sloop)=1;
    elseif ~strcmpi(szr_times{sloop,7},'Subclinical'),
        error('Unknown szr type: %s\n',szr_times{sloop,7}); 
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
    % For some reason I have to do this again. Otherwise all fields have
    % the same value regardless of file.
    file_info(floop).file_onset_sec=f_onsets(floop);
    file_info(floop).file_offset_sec=f_offsets(floop);
    file_info(floop).file_onset_hrs=file_times(floop,5);
    file_info(floop).file_dur_sec=file_dur(floop);
    clin_sonsets_this_file=[];
    clin_soffsets_this_file=[];
    clin_szr_csv_id=[];
    sub_szr_csv_id=[];
    sub_sonsets_this_file=[];
    sub_soffsets_this_file=[];
    % Loop over clinical and subclinical szrs to see if this file contains
    % any of them
    for sloop=1:n_szrs,
        if (szr_onsets(sloop)>=f_onsets(floop)) && (szr_onsets(sloop)<=f_offsets(floop))
            % Szr onset is in this file
            if (szr_offsets(sloop)>=f_onsets(floop)) && (szr_offsets(sloop)<=f_offsets(floop))
                % Szr offset is also in this file
                if clin_szr(sloop),
                    clin_sonsets_this_file=[clin_sonsets_this_file szr_onsets(sloop)-f_onsets(floop)];
                    clin_soffsets_this_file=[clin_soffsets_this_file szr_offsets(sloop)-f_onsets(floop)];
                    clin_szr_csv_id=[clin_szr_csv_id sloop];
                else
                    sub_sonsets_this_file=[sub_sonsets_this_file szr_onsets(sloop)-f_onsets(floop)];
                    sub_soffsets_this_file=[sub_soffsets_this_file szr_offsets(sloop)-f_onsets(floop)];
                    sub_szr_csv_id=[sub_szr_csv_id sloop];
                end
            else
                fprintf('Getting info from file: %s\n',file_info(floop).fname); 
                warning('Szr onset is in this file but NOT the offset. I will call all time points after onset "szr".');
                if clin_szr(sloop),
                    clin_sonsets_this_file=[clin_sonsets_this_file szr_onsets(sloop)-f_onsets(floop)];
                    clin_soffsets_this_file=[clin_soffsets_this_file f_offsets(floop)-f_onsets(floop)];
                    clin_szr_csv_id=[clin_szr_csv_id sloop];
                else
                    sub_sonsets_this_file=[sub_sonsets_this_file szr_onsets(sloop)-f_onsets(floop)];
                    sub_soffsets_this_file=[sub_soffsets_this_file f_offsets(floop)-f_onsets(floop)];
                    sub_szr_csv_id=[sub_szr_csv_id sloop];
                end
            end
        elseif (szr_offsets(sloop)>=f_onsets(floop)) && (szr_offsets(sloop)<=f_offsets(floop))
            % Szr offset is in this file but NOT the onset
            fprintf('Getting info from file: %s\n',file_info(floop).fname);
            warning('Szr offset is in this file but NOT the onset. I will call all time points before onset "szr".');
            if clin_szr(sloop),
                clin_sonsets_this_file=[clin_sonsets_this_file 0];
                clin_soffsets_this_file=[clin_soffsets_this_file szr_offsets(sloop)-f_onsets(floop)];
                clin_szr_csv_id=[clin_szr_csv_id sloop];
            else
                sub_sonsets_this_file=[sub_sonsets_this_file 0];
                sub_soffsets_this_file=[sub_soffsets_this_file szr_offsets(sloop)-f_onsets(floop)];
                sub_szr_csv_id=[sub_szr_csv_id sloop];
            end
        elseif (szr_onsets(sloop)<=f_onsets(floop)) && (szr_offsets(sloop)>=f_offsets(floop))
            % Szr spans entire file
            fprintf('Getting info from file: %s\n',file_info(floop).fname);
            warning('Szr spans entire file. I will call all time points "szr".');
            if clin_szr(sloop),
                clin_sonsets_this_file=[clin_sonsets_this_file 0];
                clin_soffsets_this_file=[clin_soffsets_this_file f_offsets(floop)-f_onsets(floop)];
                clin_szr_csv_id=[clin_szr_csv_id sloop];
            else
                sub_sonsets_this_file=[sub_sonsets_this_file 0];
                sub_soffsets_this_file=[sub_soffsets_this_file f_offsets(floop)-f_onsets(floop)];
                sub_szr_csv_id=[sub_szr_csv_id sloop];
            end
        end
    end
    file_info(floop).clin_szr_onsets_sec=clin_sonsets_this_file;
    file_info(floop).clin_szr_offsets_sec=clin_soffsets_this_file;
    file_info(floop).clin_szr_csv_id=clin_szr_csv_id;
    file_info(floop).sub_szr_onsets_sec=sub_sonsets_this_file;
    file_info(floop).sub_szr_offsets_sec=sub_soffsets_this_file;
    file_info(floop).sub_szr_csv_id=sub_szr_csv_id;
end

