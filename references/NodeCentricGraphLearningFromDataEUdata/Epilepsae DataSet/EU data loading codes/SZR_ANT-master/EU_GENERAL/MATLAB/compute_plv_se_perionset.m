%% This script computes the spectral energy (SE) and plv features with EDM for just
% a patient's clinical szrs. They are all output
% to a directory like this: SZR_ANT/EU_GENERAL/EU_GENERAL_FTRS/PLV_SE/1096/1096_HL2_HL3_szr0.mat
% 
%
%% KEY VARIABLES
% ** Voltage Data
% ieeg: raw bipolar data
% szr_class: 0=no szr, 1=clinical szr, -1=subclinical szr
% 
% targ_window: same as szr class but extends 5 sec min before clinician
% onset to try to trigger stim as early as possible
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

%*=PLV done
%sub_id=1096; %DONE*
subs=264; % DONE*
% sub_id=590; % DONE
%sub_id=253; % DONE
% subs=862; % DONE*
%subs=1125; % DONE*
% subs=1077; %DONE*
% sub_id=958;  %DONE
% sub_id=970; %DONE
% sub_id=922; %DONE
% subs=[253, 264, 590, 620, 862, 1077, 1096, 1125, 958, 970, 922];
% %subs=[273, 565]; % DONE
% subs=[115]; % DONE
% subs=[862]; % REDONE (bad szrs removed)
% subs=[922]; % REDOING (long szr relabelled as subclinical)
%subs=253; % DONE*
%subs=[273, 565, 620]; % DONE *
subs=[590, 958, 970]; %DONE *
subs=1096; % DONE *

if ismac,
    root_dir='/Users/davidgroppe/PycharmProjects/SZR_ANT/';
else
    root_dir='/home/dgroppe/GIT/SZR_ANT/';
end

for sub_id=subs,

%% Get all the files with clinical szrs and their SOZ
cli_szr_info=get_szr_fnames(sub_id);

% Remove szrs without SOZ-labels
keep_ids=[];
n_szrs=length(cli_szr_info);
fprintf('%d Clinical Szrs\n',n_szrs);
for a=1:n_szrs,
    if ~isempty(cli_szr_info(a).clinical_soz_chans{1}),
        keep_ids=[keep_ids a];
    end
end
cli_szr_info=cli_szr_info(keep_ids);
n_szrs=length(cli_szr_info);
fprintf('%d Clinical Szrs with SOZs defined\n',n_szrs);

%%
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


%% Load bad chans and ignore them
badchan_fname=fullfile(root_dir,'EU_METADATA','BAD_CHANS',sprintf('bad_chans_%d.txt',sub_id));
badchans=csv2Cell(badchan_fname);
if strcmpi(badchans{1},'None'),
    badchans=[];
end
fprintf('# of bad chans: %d\n',length(badchans));


%% Load list of channels to use for computing PLV for each electrode
plvchan_fname=fullfile(root_dir,'EU_METADATA','PLV_CHANS',sprintf('%d_plv.csv',sub_id));
plvchans=csv2Cell(plvchan_fname,',',1);
n_plv_pairs=size(plvchans,2)-1;

%% Get channel labels
bipolar_labels=derive_bipolar_pairs(sub_id);
n_chan=size(bipolar_labels,1);


%% EDM Lags and moving window lenth
edm_lags=0:2:8;
n_lags=length(edm_lags);
wind_len_sec=1; % moving window length in seconds


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


%% Get unique SOZ chans
soz_chans_mono=[];
for a=1:n_szrs,
    soz_chans_mono=union(soz_chans_mono,cli_szr_info(a).clinical_soz_chans);
end
fprintf('Monopolar SOZ chans are:\n');
disp(soz_chans_mono);

soz_chans_bi=cell(1,1);
soz_bi_ct=0;
for cloop=1:n_chan,
    use_chan=0;
    for a=1:2,
        if ismember(bipolar_labels{cloop,a},soz_chans_mono),
            use_chan=1;
        end
    end
    if use_chan,
        soz_bi_ct=soz_bi_ct+1;
        soz_chans_bi{soz_bi_ct,1}=bipolar_labels{cloop,1};
        soz_chans_bi{soz_bi_ct,2}=bipolar_labels{cloop,2};
    end
end
fprintf('Bipolar SOZ chans (Good & Bad) are:\n');
disp(soz_chans_bi);


n_soz_chan=size(soz_chans_bi,1);
good_chan_ids=[];
for a=1:n_soz_chan,
    temp_label=sprintf('%s-%s',soz_chans_bi{a,1},soz_chans_bi{a,2});
    if findStrInCell(temp_label,badchans),
        fprintf('SOZ channel %s is bad. Ignoring it.\n',temp_label);
    else
        good_chan_ids=[good_chan_ids a];
        %fprintf('%s-%s # of obs: %d\n',soz_chans_bi{a,1},soz_chans_bi{a,2},n_tpt_ct(a));
fprintf('%s-%s # of obs\n',soz_chans_bi{a,1},soz_chans_bi{a,2});
    end
end
soz_chans_bi=soz_chans_bi(good_chan_ids,:);
n_chan=size(soz_chans_bi,1);
fprintf('# of SOZ Channels (just good): %d\n',n_chan);
clear good_chan_ids


%% SGRAM Params
sgramCfg=[];
sgramCfg.Fs=256; % Sampling rate of downsampled data
sgramCfg.T=2; %1 second window
sgramCfg.K=3; % # of tapers
sgramCfg.fpass=[0 .4*sgramCfg.Fs];
W=(sgramCfg.K+1)/(2*sgramCfg.T);
sgramCfg.tapers=[W*sgramCfg.T sgramCfg.K]; %[TW K]
sgramCfg.rm_mean=0;
sgramCfg.trialave=0;
sgramCfg.movingwin=[sgramCfg.T .2];
    

%% Loop over bipolar soz chans
n_tpt_ct=zeros(size(soz_chans_bi,1),1);
for cloop=1:size(soz_chans_bi,1),
%for cloop=1:1,
    fprintf('Working on chan %s-%s\n',soz_chans_bi{cloop,1},soz_chans_bi{cloop,2});
%    for sloop=7:7,
    for sloop=1:n_szrs,
        % See if this channel was part of soz for this szr
        if ismember(soz_chans_bi{cloop,1},cli_szr_info(sloop).clinical_soz_chans) || ...
            ismember(soz_chans_bi{cloop,2},cli_szr_info(sloop).clinical_soz_chans)
            
            fprintf('Part of SOZ for Szr %d\n',sloop);
            
            % Read header
            pat=bin_file(cli_szr_info(sloop).clinical_fname);
            Fs=pat.a_samp_freq;
            if isempty(Fs),
                error('Could not find file: %s',cli_szr_info(sloop).clinical_fname);
            end
            fprintf('Fs=%f\n',Fs);
            %fprintf('# of monopolar chans %d\n',pat.a_n_chan);
            fprintf('# of samples=%d\n',(pat.a_n_samples));
            
            
            %% Figure out when szr starts/stops
            fszr_onset_tpt=round(Fs*cli_szr_info(sloop).clinical_onset_sec);
            fszr_offset_tpt=round(Fs*cli_szr_info(sloop).clinical_offset_sec);
            fprintf('Szr onset tpt %d\n',fszr_onset_tpt);
            fprintf('Szr offset tpt %d\n',fszr_offset_tpt);
            szr_class=zeros(pat.a_n_samples,1,'int8');
            % use float instead of int, easier for deubugging szr_class=zeros(pat.a_n_samples,1);
            szr_class(fszr_onset_tpt:fszr_offset_tpt)=1;
            
            % Identify target window onset for classifier 5 sec before clinician onset
            targ_window=zeros(pat.a_n_samples,1,'int8');
            % use float instead of int, easier for debugging targ_window=zeros(pat.a_n_samples,1);
            targ_onset_tpt=fszr_onset_tpt-round(Fs*5);
            if targ_onset_tpt<0,
                targ_onset_tpt=1;
            end
            
            % Identify target window offset for classifier (same as
            % clinician offset)
            targ_offset_tpt=fszr_offset_tpt;
            if targ_offset_tpt>pat.a_n_samples,
                targ_offset_tpt=pat.a_n_samples;
            end
            targ_window(targ_onset_tpt:targ_offset_tpt)=1;
            
            %%
   %         preonset_tpts=Fs*15; % 15 second preonset baseline
%             clip_onset_tpt=fszr_onset_tpt-preonset_tpts; %time pt at which to START data import
%             if clip_onset_tpt<1,
%                 clip_onset_tpt=1;
%             end
      
            
            %% Import entire clip of SOZ chan (bipolar data)
            [ieeg, time_dec, targ_raw_ieeg, targ_raw_ieeg_sec, targ_win_dec, szr_class_dec]=import_eu_clip(pat,soz_chans_bi{cloop,1},soz_chans_bi{cloop,2}, ...
                targ_window,szr_class,Fs,sgramCfg);
      
            
            %% Compute spectrogram for subsuquent visualization purposes only
            [sgram_S,sgram_t,sgram_f]=mtspecgramcDG(targ_raw_ieeg,sgramCfg.movingwin,sgramCfg);
            sgram_S=10*log10(sgram_S);
            
            
            %%
            clear targ_raw_ieeg_tpts;
            
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
            
            % Compute SE features with EDM for all data
            se_ftrs=zeros(n_lags*n_bands,n_ftr_wind);

            % Compute SE raw feature without any smoothing
            for bloop=1:n_bands,
                % Apply causal butterworth filter
                %bp_ieeg=butterfiltMK(ieeg,256,[bands(bloop,1) bands(bloop,2)],0,4);
                bp_ieeg=butterfilt4_causalEU(ieeg,256,[bands(bloop,1) bands(bloop,2)],0);
                %data=butterfilt4_causal(data,srate,flt,n_pad)
                
                % Compute moving window hilbert transform
                [se_ftrs(bloop,:), hilb_ifreq]=bp_hilb_mag(bp_ieeg,n_ftr_wind,wind_len,wind_step);
            end
            
            % Set initial value of SE features
            se_ftrs(:,1)=repmat(se_ftrs(1:n_bands,1),n_lags,1);
            
            % Apply EDM smoothing to SE features
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
            
            % Preallocate memory for PLV features with EDM
            plv_ftrs=zeros(n_lags*n_bands,n_ftr_wind);
            hilb_pcos=zeros(n_lags*n_bands,n_ftr_wind);
            hilb_psin=zeros(n_lags*n_bands,n_ftr_wind);
            
            % Get list of 7 other electrodes to compute plv with
            seed_chan=sprintf('%s-%s',soz_chans_bi{cloop,1},soz_chans_bi{cloop,2});
            plv_row_id=findStrInCell(seed_chan,plvchans(:,1));
            
            % Compute PLV features without any smoothing
            for other_chan=1:n_plv_pairs,
                fprintf('Computing PLV with chan %d of %d\n',other_chan,n_plv_pairs);
                pair_chan_bi=plvchans{plv_row_id,other_chan+1};
                dash_id=find(pair_chan_bi=='-');
                pair_chan1=pair_chan_bi(1:dash_id-1);
                pair_chan2=pair_chan_bi(dash_id+1:end);
                
                % Data from pair chan
                ieeg_pair=import_eu_clip(pat,pair_chan1,pair_chan2,targ_window,szr_class,Fs,sgramCfg);
                
                for bloop=1:n_bands,
                    % Apply causal butterworth filter to seed channel
                    bp_ieeg=butterfilt4_causalEU(ieeg,256,[bands(bloop,1) bands(bloop,2)],0);
                    
                    % Apply causal butterworth filter to pair channel
                    bp_ieeg_pair=butterfilt4_causalEU(ieeg_pair,256,[bands(bloop,1) bands(bloop,2)],0);
                    
                    % Compute PLV via moving window hilbert transform
                    [hilb_pcos(bloop,:), hilb_psin(bloop,:)]=bp_hilb_plv(bp_ieeg, bp_ieeg_pair, ...
                        n_ftr_wind,wind_len,wind_step);
                end
                
                % Apply EDM smoothing
                % Set initial value of all features
                hilb_pcos(:,1)=repmat(hilb_pcos(1:n_bands,1),n_lags,1);
                hilb_psin(:,1)=repmat(hilb_psin(1:n_bands,1),n_lags,1);
                
                fprintf('Applying EDM smoothing to PLV cos & sin...\n');
                ftr_ids=1:n_bands;
                base_ftr_ids=1:n_bands;
                for edm_loop=2:n_lags,
                    ftr_ids=ftr_ids+n_bands;
                    now_wt=1/(2^edm_lags(edm_loop));
                    for t=2:n_ftr_wind,
                        hilb_pcos(ftr_ids,t)=now_wt*hilb_pcos(base_ftr_ids,t)+(1-now_wt)*hilb_pcos(ftr_ids,t-1);
                        hilb_psin(ftr_ids,t)=now_wt*hilb_psin(base_ftr_ids,t)+(1-now_wt)*hilb_psin(ftr_ids,t-1);
                    end
                end
                plv_ftrs=plv_ftrs+sqrt(hilb_pcos.*hilb_pcos+hilb_psin.*hilb_psin)/n_plv_pairs;
            end
            
            
            % Remove initial time points polluted by edge effects
            se_ftrs=se_ftrs(:,edge_pts:end);
            plv_ftrs=plv_ftrs(:,edge_pts:end);
            se_time_sec=se_time_sec(edge_pts:end);
            se_class=se_class(edge_pts:end);
            se_szr_class=se_szr_class(edge_pts:end);
            
            if 0,
                % Code for checking results
                se_ftrs_z=zscore(se_ftrs')'; %z-score
                plv_ftrs_z=zscore(plv_ftrs')'; %z-score
                
                figure(1); clf();
                ax1=subplot(411);
                %imagesc(se_ftrs_z);
                h=plot(se_time_sec,se_ftrs_z);
                title(sprintf('%s-%s, Szr%d',soz_chans_bi{cloop,1},soz_chans_bi{cloop,2},sloop));
                
                ax2=subplot(412); plot(se_time_sec,se_class,'--b'); hold on;
                plot(se_time_sec,se_szr_class,'r-');
                axis tight;
                
                ax3=subplot(413);
                plot(se_time_sec,plv_ftrs_z);
                axis tight;
                
                ax4=subplot(414);
                plot(time_dec,ieeg);
                axis tight;
                
                linkaxes([ax1 ax2 ax3 ax4],'x');
            end
            
            %% Grab only features during target window
            se_targ_ids=find(se_class>=0.5);
            se_ftrs=se_ftrs(:,se_targ_ids);
            plv_ftrs=plv_ftrs(:,se_targ_ids);
            se_time_sec=se_time_sec(se_targ_ids);
            se_class=se_class(se_targ_ids);
            se_szr_class=se_szr_class(se_targ_ids);
            
            n_tpt_ct(cloop)=n_tpt_ct(cloop)+length(se_targ_ids); % Keep track
            % of how many target observations were captured for each
            % electrode so that we can sample an equal number of non-target
            % examples
            
            %% Save just target window time points
            outdir=fullfile(root_dir,'EU_GENERAL','EU_GENERAL_FTRS','PLV_SE', ...
                num2str(sub_id));
            if ~exist(outdir,'dir'),
               mkdir(outdir); 
            end
            outfname=fullfile(outdir,sprintf('%d_%s_%s_szr%d',sub_id, ...
                soz_chans_bi{cloop,1},soz_chans_bi{cloop,2}, ...
                cli_szr_info(sloop).clinical_szr_num));
            szr_fname=cli_szr_info(sloop).clinical_fname;
            fprintf('Saving szr features to %s\n',outfname);
            save(outfname,'se_ftrs','se_time_sec','se_szr_class','plv_ftrs', ...
                'ftr_labels','szr_fname','targ_raw_ieeg','targ_raw_ieeg_sec', ...
                'sgram_S','sgram_t','sgram_f');
        end
        
    end
end


%%  Save sample size of each electrode
ftr_fs=1/median(diff(se_time_sec));
outdir=fullfile(root_dir,'EU_GENERAL','EU_GENERAL_FTRS');
outfname=fullfile(outdir,sprintf('%d_szr_sample_size',sub_id));
fprintf('Saving counts of # of szr observations/electrode in %s\n',outfname);
save(outfname,'n_tpt_ct','soz_chans_bi','ftr_fs');

disp('Done!!');

end
