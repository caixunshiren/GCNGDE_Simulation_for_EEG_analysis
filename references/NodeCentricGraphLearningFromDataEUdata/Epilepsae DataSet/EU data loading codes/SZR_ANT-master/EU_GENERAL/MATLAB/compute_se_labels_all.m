%% This script computes just the ictal and target windows for ALL 
% of a patient's clips.  They are all output to a directory like
% this: /home/dgroppe/EU_Y/264_all_labels'
%
% KEY VARIABLES
% ** Voltage Data
% ieeg: raw bipolar data
% szr_class: 0=no szr, 1=szr (clinical or otherwise)
% 
% targ_window: same as szr class but extends 10 min before and after to deal with noise in onset/offset
% definition
% 
% ieeg_time_sec_pre_decimate: time relative to start of file
% 
% 
% ** Downsampled Data (256 Hz for all patients)
% ieeg: raw bipolar data
% time_dec: dowsample dtime
% targ_wind_dec=downsampled version of targ_window
% szr_class_dec=downsampled version of szr_class
% 
% 
% ** Spectral Energy Features
% n_ftr_wind: # of feature time pts (about 10 Hz srate)
% se_time_sec: time relative to start of file
% se_szr_class: 0=no szr, 1=szr
% se_class: 0=no szr, 1=szr <-target windowfor classifier
% se_ftrs: feature matrix (ftr x time)
% ftr_labes: feature labels



%% Choose Patient
subs=[115, 264, 273, 862, 1125];
subs=1096;  %done Sept 2018
subs=1077; %done Setp 2018
subs=970; %done Sept 2018
subs=253; %done Sept 2018
subs=565; %done Sept 2018
subs=590; %done Sept 2018
subs=[620, 958];


for sub_id=subs,

if ismac,
    root_dir='/Users/davidgroppe/PycharmProjects/SZR_ANT/';
    outdir=fullfile('/Volumes/SgateOSExJnld/EU_TEMP/',[num2str(sub_id) '_all_labels']);
else
    root_dir='/home/dgroppe/GIT/SZR_ANT/';
    outdir=fullfile('/home/dgroppe/EU_Y/',[num2str(sub_id) '_all_labels']);
end


%% Get list of all files and the timing of all szrs (clinical and subclinical)
%file_info=get_fnames_and_szr_times2(sub_id);
file_info=get_fnames_and_szr_times(sub_id);
n_files=length(file_info);

% outdir=fullfile(root_dir,'EU_GENERAL','EU_GENERAL_FTRS','SE');
if ~exist(outdir,'dir'),
    mkdir(outdir);
end

wind_len_sec=1; % moving window length in seconds
edge_pts=1178; % # of initial time points to remove due to edge effect


%% Loop over files
%for floop=1:2, %PICKUP ?? start at n_files=25
for floop=1:n_files,
    % Compute features
    
    % Read header
    full_data_fname=get_data_fullfname(sub_id,file_info(floop).fname);
    fprintf('Loading file %d/%d %s\n',floop,n_files,file_info(floop).fname);
    pat=bin_file(full_data_fname);
    Fs=pat.a_samp_freq;
    if isempty(Fs),
        error('Could not find file: %s',full_data_fname);
    end
    fprintf('Fs=%f\n',Fs);
    fprintf('# of samples=%d\n',(pat.a_n_samples));
    
    % Import entire clip (typically 1 hour long)
    ieeg=zeros(pat.a_n_samples,1);
    ieeg_time_sec_pre_decimate=[0:(length(ieeg)-1)]/Fs; % time relative to start of file
    

    
    %% Compute ictal-class
    szr_class=zeros(1,length(ieeg)); % -1=subclinical szr, 1=clinical szr
    targ_window=zeros(1,length(ieeg)); % Same as szr class but extended 
    % 5 seconds before onset to try to stimulate before onset
    
    % Clinical Szrs
    if ~isempty(file_info(floop).clin_szr_onsets_sec),
        % There are szrs in this file (clinical and/or subclinical)
        for sloop=1:length(file_info(floop).clin_szr_onsets_sec),
            onset_id=findTpt(file_info(floop).clin_szr_onsets_sec(sloop),ieeg_time_sec_pre_decimate);
            if ~isempty(file_info(floop).clin_szr_offsets_sec),
                % Sadly, some szrs have marked onsets but not offsets
                % When this happens make szr last until end of clip
                offset_id=findTpt(file_info(floop).clin_szr_offsets_sec(sloop),ieeg_time_sec_pre_decimate);
            else
                offset_id=length(ieeg);
            end
            szr_class(onset_id:offset_id)=1;
            targ_onset_id=onset_id-Fs*5; %extend 5 seconds in past to try to stimulate before onset
            if targ_onset_id<1,
                targ_onset_id=1;
            end
            targ_window(targ_onset_id:offset_id)=1;
        end
    end
    
    % Subclinical Szrs
    if ~isempty(file_info(floop).sub_szr_onsets_sec),
        % There are szrs in this file (clinical and/or subclinical)
        for sloop=1:length(file_info(floop).sub_szr_onsets_sec),
            onset_id=findTpt(file_info(floop).sub_szr_onsets_sec(sloop),ieeg_time_sec_pre_decimate);
            if ~isempty(file_info(floop).sub_szr_offsets_sec),
                % Sadly, some szrs have marked onsets but not offsets
                % When this happens make szr last until end of clip
                offset_id=findTpt(file_info(floop).sub_szr_offsets_sec(sloop),ieeg_time_sec_pre_decimate);
            else
                offset_id=length(ieeg);
            end
            szr_class(onset_id:offset_id)=-1;
            targ_onset_id=onset_id-Fs*5; %extend 5 seconds in past to try to stimulate before onset
            if targ_onset_id<1,
                targ_onset_id=1;
            end
            targ_window(targ_onset_id:offset_id)=-1;
        end
    end
    
    
    %% Downsample if necessary
    if Fs>256,
        % Downsample data to 256 Hz
        down_fact=round(Fs/256);
        ieeg=decimate(ieeg,down_fact);
        time_dec=zeros(1,length(ieeg));
        szr_class_dec=zeros(1,length(ieeg));
        for tloop=1:length(ieeg),
            time_dec(tloop)=mean(ieeg_time_sec_pre_decimate([1:down_fact] ...
                +(tloop-1)*down_fact));
            szr_class_dec(tloop)=mean(szr_class([1:down_fact] ...
                +(tloop-1)*down_fact));
        end
    else
        time_dec=ieeg_time_sec_pre_decimate;
        szr_class_dec=szr_class;
    end
    clear ieeg_time_sec_pre_decimate;
    
    %% Compute features
    % Figure out how many feature time points there are
    
    wind_len=256*wind_len_sec;
    wind_step=round(256/10); %10 Hz moving window sampling
    wind=1:wind_len;
    n_ftr_wind=0;
    while wind(end)<length(ieeg),
        n_ftr_wind=n_ftr_wind+1;
        wind=wind+wind_step;
    end
    n_ftr_wind=n_ftr_wind-1;
    
    % Find mean time and class of moving windows
    se_time_sec=zeros(1,n_ftr_wind);
    se_szr_class=zeros(1,n_ftr_wind);
    wind=1:wind_len;
    for a=1:n_ftr_wind,
        se_time_sec(a)=mean(time_dec(wind));
        se_szr_class(a)=mean(szr_class_dec(wind));
        wind=wind+wind_step;
    end

    % Remove initial time points polluted by edge effects
    se_szr_class=se_szr_class(edge_pts:end); % Whether or not time window is within clinician defined onset-offset
    se_time_sec=se_time_sec(edge_pts:end); % Time in second that corresponds to se_szr_class labels
    
    % Save results to disk
    temp_id=find(file_info(floop).fname=='.');
    fname_stem=file_info(floop).fname(1:temp_id-1);
    outfname=fullfile(outdir,sprintf('%d_y_%s',sub_id,fname_stem));
    fprintf('Saving szr features to %s\n',outfname);
    file_onset_sec=file_info(floop).file_onset_sec; % Onset of raw ieeg in seconds relative to anchor date (Jan 1, 2000 I think)
    save(outfname,'se_szr_class','se_time_sec','szr_class_dec','time_dec', ...
        'file_onset_sec');
end
end

disp('Done!!');

