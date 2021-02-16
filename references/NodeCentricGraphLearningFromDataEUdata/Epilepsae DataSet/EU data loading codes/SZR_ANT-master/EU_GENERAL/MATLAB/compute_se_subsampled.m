%% This script computes the spectral energy (SE) features with EDM for a
% random subset of a patient's data. These values are used for z-scoring
% labeled data for training.
% They are all output to a directory like this:
%  SZR_ANT/EU_GENERAL/EU_GENERAL_FTRS/SE/1096/1096_HL2_HL3_subsamp.mat
% 
%
%% KEY VARIABLES
% ** Voltage Data
% ieeg: raw bipolar data
% szr_class: 0=no szr, 1=clinical szr, -1=subclinical szr
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
% ftr_labels: feature labels


% DONE
% 264, 273, 565, 590, 1096, 1125
%sub_id=620; 
%sub_id=264; % DONE
% sub_id=590; 
% sub_id=862;
% sub_id=565; 
%sub_id=253; 
%sub_id=273; % DONE
%sub_id=1125; % DONE
%sub_id=1077; 

if ismac,
    root_dir='/Users/davidgroppe/PycharmProjects/SZR_ANT/';
else
    root_dir='/home/dgroppe/GIT/SZR_ANT/';
end
% DONE subs=[253, 620, 862, 1077];
% DONE subs=[958, 970];
% DONE subs=[922];
% DONE subs=[264];
subs=[115];
subs=442;
subs=[862];
%subs=922; %DONE


for sub_id=subs,

%% EDM Lags and moving window length
edm_lags=0:2:8;
n_lags=length(edm_lags);
wind_len_sec=1; % moving window length in seconds

edge_pts=1178; % # of initial time points to remove due to edge effect
% Decay fact 2, it takes 18 tpts (1.800000 sec) to get below .01 threshold
% Decay fact 4, it takes 73 tpts (7.300000 sec) to get below .01 threshold
% Decay fact 6, it takes 294 tpts (29.400000 sec) to get below .01 threshold
% Decay fact 8, it takes 1178 tpts (117.800000 sec) to get below .01 threshold
% Decay fact 2, it takes 18 tpts (1.800000 sec) to get below .01 threshold
% Decay fact 4, it takes 73 tpts (7.300000 sec) to get below .01 threshold
% Decay fact 6, it takes 294 tpts (29.400000 sec) to get below .01 threshold
% Decay fact 8, it takes 1178 tpts (117.800000 sec) to get below .01 threshold
% Decay fact 10, it takes 4715 tpts (471.500000 sec) to get below .01 threshold


%% Define frequency bands
bands=[0 4; 4 8; 8 13; 13 30; 30 50; 70 100];
band_labels={'DeltaMag','ThetaMag','AlphaMag','BetaMag','GammaMag','HGammaMag'};
n_bands=length(bands);

%% Derive feature labels
n_ftrs=n_bands*n_lags;
ftr_labels=cell(n_ftrs,1);
temp_ct=0;
for lag_loop=1:n_lags,
    for band_loop=1:n_bands,
        temp_ct=temp_ct+1;
        ftr_labels{temp_ct}=sprintf('%s_Lag_%d',band_labels{band_loop},edm_lags(lag_loop));
    end
end

%% Load bad chans and ignore them
badchan_fname=fullfile(root_dir,'EU_METADATA','BAD_CHANS',sprintf('bad_chans_%d.txt',sub_id));
badchans=csv2Cell(badchan_fname);
if strcmpi(badchans{1},'None'),
    badchans=[];
end
fprintf('# of bad chans: %d\n',length(badchans));


%% Load list of SOZ channels and the number of samples for each
indir=fullfile(root_dir,'EU_GENERAL','EU_GENERAL_FTRS');
infname=fullfile(indir,sprintf('%d_szr_sample_size',sub_id));
fprintf('Loading counts of # of szr observations/electrode in %s\n',infname);
% save(infname,'n_tpt_ct','soz_chans_bi','ftr_fs');
load(infname);

n_chan=size(soz_chans_bi,1);
fprintf('# of SOZ Channels (good & bad): %d\n',n_chan);
fprintf('Feature sampling rate: %f Hz\n',ftr_fs);
good_chan_ids=[];
for a=1:n_chan,
    temp_label=sprintf('%s-%s',soz_chans_bi{a,1},soz_chans_bi{a,2});
    if findStrInCell(temp_label,badchans),
        fprintf('SOZ channel %s is bad. Ignoring it.\n',temp_label);
    else
        good_chan_ids=[good_chan_ids a];
        fprintf('%s-%s # of obs: %d\n',soz_chans_bi{a,1},soz_chans_bi{a,2},n_tpt_ct(a));
    end
end
soz_chans_bi=soz_chans_bi(good_chan_ids,:);
n_chan=size(soz_chans_bi,1);
fprintf('# of SOZ Channels (just good): %d\n',n_chan);
clear good_chan_ids


%% Get list of all files and the timing of all szrs (clinical and subclinical)
file_info=get_fnames_and_szr_times(sub_id);
n_files=length(file_info);


%% Loop over SOZ electrodes
%for cloop=1:1,
for cloop=1:n_chan, 
    fprintf('Working on chan %s-%s\n',soz_chans_bi{cloop,1},soz_chans_bi{cloop,1});
    
    % Figure out how many non-szr samples to draw from each file
    n_subsamp_files=round(n_files/2); % only sample for the first half of files since in
    % practice we will only have files from beginning of patient stay to
    % estimate normalization
    n_subsamp_obs=ceil(n_tpt_ct(cloop)/n_subsamp_files);
    fprintf('Drawing %d random samples (szr and non-szr) from each file.\n',n_subsamp_obs);
    
    subsamp_se_ftrs=[];
    % Loop over first half of files
    for floop=1:n_subsamp_files,
        % Compute features
        
        % Read header
        full_data_fname=get_data_fullfname(sub_id,file_info(floop).fname);
        fprintf('Channel %d/%d\n',cloop,n_chan);
        fprintf('Loading file %d/%d %s\n',floop,n_subsamp_files,file_info(floop).fname);
        pat=bin_file(full_data_fname);
        Fs=pat.a_samp_freq;
        if isempty(Fs),
            error('Could not find file: %s',full_data_fname);
        end
        fprintf('Fs=%f\n',Fs);
        %fprintf('# of monopolar chans %d\n',pat.a_n_chan);
        fprintf('# of samples=%d\n',(pat.a_n_samples));
        
        % Import entire clip (typically 1 hour long)
        pat.a_channs_cell={soz_chans_bi{cloop,1}}; % Channel to import
        ieeg_temp1=pat.get_bin_signals(1,pat.a_n_samples);
        
        pat.a_channs_cell={soz_chans_bi{cloop,2}}; % Channel to import
        ieeg_temp2=pat.get_bin_signals(1,pat.a_n_samples);
        
        ieeg=ieeg_temp1-ieeg_temp2;
        ieeg_time_sec_pre_decimate=[0:(length(ieeg)-1)]/Fs; % time relative to start of file
        clear ieeg_temp1 ieeg_temp2;
        
        % Compute ictal-class
        szr_class=zeros(1,length(ieeg));
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
        
        % Downsample if necessary
        if Fs>256,
            % Downsample data to 256 Hz
            down_fact=round(Fs/256);
            ieeg=decimate(ieeg,down_fact);
            time_dec=zeros(1,length(ieeg));
            targ_win_dec=zeros(1,length(ieeg));
            szr_class_dec=zeros(1,length(ieeg));
            for tloop=1:length(ieeg),
                time_dec(tloop)=mean(ieeg_time_sec_pre_decimate([1:down_fact] ...
                    +(tloop-1)*down_fact));
                targ_win_dec(tloop)=mean(targ_window([1:down_fact] ...
                    +(tloop-1)*down_fact));
                szr_class_dec(tloop)=mean(szr_class([1:down_fact] ...
                    +(tloop-1)*down_fact));
            end
        else
            time_dec=ieeg_time_sec_pre_decimate;
            targ_win_dec=targ_window;
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
        se_class=zeros(1,n_ftr_wind);
        se_szr_class=zeros(1,n_ftr_wind);
        wind=1:wind_len;
        for a=1:n_ftr_wind,
            se_time_sec(a)=mean(time_dec(wind));
            se_class(a)=mean(targ_win_dec(wind));
            se_szr_class(a)=mean(szr_class_dec(wind));
            wind=wind+wind_step;
        end
        
        % Compute features with EDM for all data
        se_ftrs=zeros(n_lags*n_bands,n_ftr_wind);
        
        % Compute raw feature without any smoothing
        for bloop=1:n_bands,
            % Apply causal butterworth filter
            %bp_ieeg=butterfiltMK(ieeg,256,[bands(bloop,1) bands(bloop,2)],0,4);
            bp_ieeg=butterfilt4_causalEU(ieeg,256,[bands(bloop,1) bands(bloop,2)],0);
            
            % Compute moving window hilbert transform
            [se_ftrs(bloop,:), hilb_ifreq]=bp_hilb_mag(bp_ieeg,n_ftr_wind,wind_len,wind_step);
        end
        
        % Set initial value of all features
        se_ftrs(:,1)=repmat(se_ftrs(1:n_bands,1),n_lags,1);
        
        % Apply EDM smoothing
        fprintf('Applying EDM smoothing...\n');
        ftr_ids=1:n_bands;
        base_ftr_ids=1:n_bands;
        for edm_loop=2:n_lags,
            ftr_ids=ftr_ids+n_bands;
            now_wt=1/(2^edm_lags(edm_loop));
            for t=2:n_ftr_wind,
                se_ftrs(ftr_ids,t)=now_wt*se_ftrs(base_ftr_ids,t)+(1-now_wt)*se_ftrs(ftr_ids,t-1);
            end
        end
        
        % Remove initial time points polluted by edge effects
        se_ftrs=se_ftrs(:,edge_pts:end);
        se_time_sec=se_time_sec(edge_pts:end);
        se_class=se_class(edge_pts:end);
        se_szr_class=se_szr_class(edge_pts:end);
        
        if isempty(subsamp_se_ftrs),
            %preallocate mem, first time adding data
            subsamp_se_ftrs=zeros(n_ftrs,n_subsamp_obs*n_subsamp_files);
            subsamp_se_ftrs_time_sec=zeros(1,n_subsamp_obs*n_subsamp_files);
            source_fnames=cell(1,n_subsamp_files);
            subsamp_szr_class=zeros(1,n_subsamp_obs*n_subsamp_files);
            subsamp_targ_class=zeros(1,n_subsamp_obs*n_subsamp_files);
        end
        
        % Get non-szr time window ids
        n_post_edge_ftr_wind=length(se_class);
%         temp_time_ids=1:n_post_edge_ftr_wind;
        % Randomly select subset of features (both szr and non szr
        if n_post_edge_ftr_wind>=n_subsamp_obs,
            subsamp_ids=randi(n_post_edge_ftr_wind,1,n_subsamp_obs);
            subsamp_se_ftrs(:,[1:n_subsamp_obs]+(floop-1)*n_subsamp_obs)=se_ftrs(:,subsamp_ids);
            subsamp_se_ftrs_time_sec([1:n_subsamp_obs]+(floop-1)*n_subsamp_obs)=se_time_sec(subsamp_ids);
            subsamp_szr_class([1:n_subsamp_obs]+(floop-1)*n_subsamp_obs)=se_szr_class(subsamp_ids);
            subsamp_targ_class([1:n_subsamp_obs]+(floop-1)*n_subsamp_obs)=se_class(subsamp_ids);
        else
            % Not enought data points in file, fill with NaN values
            subsamp_se_ftrs(:,[1:n_subsamp_obs]+(floop-1)*n_subsamp_obs)=NaN;
            subsamp_se_ftrs_time_sec([1:n_subsamp_obs]+(floop-1)*n_subsamp_obs)=NaN;
            subsamp_szr_class([1:n_subsamp_obs]+(floop-1)*n_subsamp_obs)=NaN;
            subsamp_targ_class([1:n_subsamp_obs]+(floop-1)*n_subsamp_obs)=NaN;
        end
        source_fnames{floop}=full_data_fname;

        
    end
    
    %% Remove any nan values
    non_nan_ids=find(~isnan(subsamp_se_ftrs_time_sec));
    subsamp_se_ftrs=subsamp_se_ftrs(:,non_nan_ids);
    subsamp_se_ftrs_time_sec=subsamp_se_ftrs_time_sec(non_nan_ids);
    subsamp_szr_class=subsamp_szr_class(non_nan_ids);
    subsamp_targ_class=subsamp_targ_class(non_nan_ids);
    
    %% Save results to disk
    outdir=fullfile(root_dir,'EU_GENERAL','EU_GENERAL_FTRS','SE',num2str(sub_id));
    if ~exist(outdir,'dir'),
        mkdir(outdir);
    end
    outfname=fullfile(outdir,sprintf('%d_%s_%s_subsamp',sub_id, ...
        soz_chans_bi{cloop,1},soz_chans_bi{cloop,2}));
    fprintf('Saving szr features to %s\n',outfname);
%     non_nan_ids=find(~isnan(subsamp_se_ftrs_time_sec));
%     subsamp_se_ftrs_time_sec=subsamp_se_ftrs_time_sec(non_nan_ids);
%     subsamp_se_ftrs=subsamp_se_ftrs(:,non_nan_ids);
    save(outfname,'subsamp_se_ftrs','subsamp_se_ftrs_time_sec','source_fnames', ...
        'ftr_labels','subsamp_szr_class','subsamp_targ_class');
end


end



disp('Done!!');

