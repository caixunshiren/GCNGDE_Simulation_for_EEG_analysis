% This script:
% 1. Loads 7 channels of eeg data files that contain seizures
% (both clinical and subclinical)
% 2. Decimates data to 256 Hz
% 3. Converts data to avg ref and adds the mean as 8th channel
% 4. Identifies which time points contain szrs
% 5. Writes data to a mat file for use in Python

sub=1096;
times_fname=fullfile('/Users/dgroppe/Desktop/Dropbox/TWH_INFO/EU_METADATA/',['data_on_off_FR_' num2str(sub) '.csv']);
fprintf('Reading *.data onset/offset times from %s\n',times_fname);
times_csv=csv2Cell(times_fname,',',1);


%% Load seizure times in seconds
szr_times_fname=fullfile('/Users/dgroppe/Desktop/Dropbox/TWH_INFO/EU_METADATA/',['szr_on_off_FR_' num2str(sub) '.csv']);
szr_csv=csv2Cell(szr_times_fname,',',1);
n_szrs=size(szr_csv,1);
szr_onoff_sec=zeros(n_szrs,2);
for a=1:n_szrs,
    szr_onoff_sec(a,1)=str2num(szr_csv{a,4});
    szr_onoff_sec(a,2)=str2num(szr_csv{a,2});
end


%% Convert *.header names to *.dat and file times to seconds
n_hdr_files=size(times_csv,1);
csv_data_fnames=cell(n_hdr_files,1);
file_times_sec=zeros(n_hdr_files,2);
for a=1:n_hdr_files,
   tmp=times_csv{a,3};
   dot_id=find(tmp=='.');
   csv_data_fnames{a}=[tmp(1:dot_id-1) '.data'];
   file_times_sec(a,1)=str2num(times_csv{a,4});
   file_times_sec(a,2)=str2num(times_csv{a,6});
end


%% Figure out which files have seizures in them
szr_files=[];
szr_data_file_ids=zeros(n_szrs,1);
for a=1:n_szrs,
    pre_ids=find(file_times_sec(:,1)<szr_onoff_sec(a,1));
    post_ids=find(file_times_sec(:,2)>szr_onoff_sec(a,2));
    contains_id=intersect(pre_ids,post_ids);
    if length(contains_id)>1,
        error('Szr occurs in more than on *.data file. Bookeeping error.\n');
    end
    szr_data_file_ids(a)=contains_id;
    fprintf('Szr#%d occurs in *.data file %d, %s\n',a,contains_id,csv_data_fnames{contains_id});
end

%% Channels to ignore:
% ignore_channels={'TP1','TP2','TP3','TP4','FP1','FP2','F3','F4','F7','F8','C1','C2','C3','C4', ...
%     'P3','P4','O1','O2','T1','T2','T3','T4','T5','T6','FZ','CZ','PZ','EMG','ECG','EOG1','EOG2'};
% bad_chan_fname=fullfile('/Users/dgroppe/Desktop/Dropbox/TWH_INFO/EU_METADATA/',['bad_chans_FR_' num2str(sub) '.txt']);
% fprintf('Reading bad channel names from %s\n',bad_chan_fname);
% bad_chans=csv2Cell(bad_chan_fname)';
% ignore_channels=[bad_chans, ignore_channels];


%% Get list of electrodes to import
elec_path='/Users/dgroppe/Desktop/Dropbox/TWH_INFO/EU_INFO';
in_fname=fullfile(elec_path,'FR_1096_use_elecs.txt');
use_chans=csv2Cell(in_fname);
fprintf('Using the following %d channels\n',length(use_chans));
for a=1:length(use_chans),
   fprintf('%s\n',use_chans{a}); 
end
chan_labels=use_chans;
chan_labels{8}='Mean';


%% Get list of all files for a patient
in_data_path='/Volumes/ValianteLabEuData/EU/inv/pat_FR_1096/adm_1096102/rec_109600102/';
fdir=dir(fullfile(in_data_path,'*.data'));
n_raw_files=length(fdir);
fprintf('%d total *.data files\n',n_raw_files);
file_ct=0;


%% Loop over files
out_data_path='/Users/dgroppe/EU_MAT_DATA/FR_1096';
if ~exist(out_data_path,'dir'),
    fprintf('Creating dir %s\n',out_data_path);
    mkdir(out_data_path);
end

%for floop=1:5, 
for floop=1:n_hdr_files,
    just_ieeg_fname=fdir(floop).name;
    ieeg_fname=fullfile(in_data_path,just_ieeg_fname);
    fprintf('Reading %s\n',ieeg_fname);
    
    % Read header
    pat=bin_file(ieeg_fname);
    Fs=pat.a_samp_freq;
    fprintf('FS=%f\n',Fs);
    fprintf('# of chans %d\n',pat.a_n_chan);
    fprintf('# of samples=%d\n',(pat.a_n_samples));
    ieeg=zeros(8,pat.a_n_samples);
    %     disp(pat.a_file_elec_cell);
    %     disp(pat.a_start_ts);
    %     disp(pat.a_stop_ts);

    % Import just channels of interest
    %pat.a_channs_cell=setdiff(pat.a_file_elec_cell,ignore_channels); % Remove none iEEG channels
    %use_chans=pat.a_channs_cell;
    pat.a_channs_cell=use_chans; % Channels to import
    ieeg(1:7,:)=pat.get_bin_signals([],[]);
    % Convert to avg ref and save avg as 8th channel
    ieeg(8,:)=mean(ieeg(1:7,:));
    for chan_loop=1:7,
       ieeg(chan_loop,:)=ieeg(chan_loop,:)-ieeg(8,:);
    end
    
    % Get timing from csv file
    csv_id=findStrInCell(just_ieeg_fname,csv_data_fnames,1);
    start_sec=str2num(times_csv{csv_id,4});
    stop_sec=str2num(times_csv{csv_id,6});
    
    % Downsample to 256 Hz
    div_fact=Fs/256;
    % dec_fact=round(log2(div_fact));
    n_tpt=round(pat.a_n_samples/div_fact);
    n_chan=size(ieeg,1);
    ieeg256=zeros(n_chan,n_tpt);
    for c=1:n_chan,
        fprintf('Decimating chan %d/%d\n',c,n_chan);
        ieeg256(c,:)=decimate(ieeg(c,:),div_fact);
    end
%         tpts_sec=start_sec:(1/Fs):(stop_sec-1/Fs);
%     tpts_sec=decimate(tpts_sec,div_fact);
    Fs=round(Fs/div_fact);
    tpts_sec=start_sec:(1/Fs):(stop_sec-1/Fs);
    clear ieeg
    
    is_szr=zeros(length(tpts_sec),1,'int8');
    szr_type='NoSzr';
    tmp_szr_file_id=find(floop==szr_data_file_ids);
    if ~isempty(tmp_szr_file_id),
        % This segment of data contains a seizure
        fprintf('This file contains %d seizure(s).\n',length(tmp_szr_file_id));
        for sloop=1:length(tmp_szr_file_id),
            szr_ids=find( (tpts_sec>=szr_onoff_sec(tmp_szr_file_id(sloop),1)).* ...
                (tpts_sec<=szr_onoff_sec(tmp_szr_file_id(sloop),2))); % all time points during szr
            is_szr(szr_ids)=1;
            if sloop==1,
                szr_type=szr_csv{tmp_szr_file_id(sloop),6};
            else
                szr_type=[szr_type ', ' szr_csv{tmp_szr_file_id(sloop),6}];
            end
        end
    else
        fprintf('This file does NOT contain a seizure.\n');
    end
            
    % Save
    start_ts=pat.a_start_ts;
    stop_ts=pat.a_stop_ts;
    dot_id=find(just_ieeg_fname=='.');
    stem=just_ieeg_fname(1:dot_id-1);
    out_fname=[stem '.mat'];
    fprintf('Saving file to %s\n',fullfile(out_data_path,out_fname));
    save(fullfile(out_data_path,out_fname),'Fs','tpts_sec','start_ts','stop_ts','ieeg256','is_szr','chan_labels','szr_type','just_ieeg_fname');
    clear ieeg256 tpts_sec is_szr
end

%%
fprintf('ALLLLLLLL DONE!!!!!!!!!!!\n');