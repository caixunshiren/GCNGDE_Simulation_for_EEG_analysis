%sub_id=1096; %DONE
 sub_id=620;
%sub_id=264;
% sub_id=590;
%sub_id=253;
% sub_id=862;
% sub_id=565;
if ismac,
    root_dir='/Users/davidgroppe/PycharmProjects/SZR_ANT/';
else
    root_dir='/home/dgroppe/GIT/SZR_ANT/';
end


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
fprintf('Bipolar SOZ chans are:\n');
disp(soz_chans_bi);


%% Loop over bipolar soz chans
n_tpt_ct=zeros(size(soz_chans_bi,1),1);
for cloop=1:size(soz_chans_bi,1),
%for cloop=1:1,
    fprintf('Working on chan %s-%s\n',soz_chans_bi{cloop,1},soz_chans_bi{cloop,2});
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
            fprintf('FS=%f\n',Fs);
            %fprintf('# of monopolar chans %d\n',pat.a_n_chan);
            fprintf('# of samples=%d\n',(pat.a_n_samples));
            
            
            %%
            fszr_onset_tpt=round(Fs*cli_szr_info(sloop).clinical_onset_sec);
            fszr_offset_tpt=round(Fs*cli_szr_info(sloop).clinical_offset_sec);
            fprintf('Szr onset tpt %d\n',fszr_onset_tpt);
            fprintf('Szr offset tpt %d\n',fszr_offset_tpt);
            szr_class=zeros(pat.a_n_samples,1,'int8');
            szr_class(fszr_onset_tpt:fszr_offset_tpt)=1;
            
            % Identify target window onset for classifier 5 sec before clinician onset
            targ_window=zeros(pat.a_n_samples,1,'int8');
            targ_onset_tpt=fszr_onset_tpt-round(Fs*5);
            if targ_onset_tpt<0,
                targ_onset_tpt=1;
            end
            
            % Identify target window offset for classifier 10 sec after clinician onset
            %             targ_offset_tpt=fszr_onset_tpt+round(Fs*10);
            %             if targ_offset_tpt>pat.a_n_samples,
            %                 targ_offset_tpt=pat.a_n_samples;
            %             end
            %             targ_window(targ_onset_tpt:targ_offset_tpt)=1;
            
            % Identify target window offset for classifier=clinician offset
            targ_offset_tpt=fszr_offset_tpt;
            if targ_offset_tpt>pat.a_n_samples,
                targ_offset_tpt=pat.a_n_samples;
            end
            targ_window(targ_onset_tpt:targ_offset_tpt)=1;
            
            %%
            preonset_tpts=Fs*15; % 15 second preonset baseline
            clip_onset_tpt=fszr_onset_tpt-preonset_tpts; %time pt at which to START data import
            if clip_onset_tpt<1,
                clip_onset_tpt=1;
            end
            
            % Import entire clip (typically 1 hour long)
            %             ieeg_labels=cell(n_chan,1);
            pat.a_channs_cell={soz_chans_bi{cloop,1}}; % Channel to import
            %ieeg(1:n_chan,:)=pat.get_bin_signals([],[]);
            ieeg_temp1=pat.get_bin_signals(1,pat.a_n_samples);
            
            pat.a_channs_cell={soz_chans_bi{cloop,2}}; % Channel to import
            ieeg_temp2=pat.get_bin_signals(1,pat.a_n_samples);
            
            ieeg=ieeg_temp1-ieeg_temp2;
            ieeg_time_sec_pre_decimate=[0:(length(ieeg)-1)]/Fs; % time relative to start of file
            clear ieeg_temp1 ieeg_temp2;
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
            
            % Clip raw szr data
            targ_raw_ieeg_tpts=(targ_win_dec>0);
            targ_raw_ieeg=ieeg(targ_raw_ieeg_tpts);
            targ_raw_ieeg_sec=time_dec(targ_raw_ieeg_tpts);
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
            
            % Compute features with EDM for all data
            se_ftrs=zeros(n_lags*n_bands,n_ftr_wind);

            % Compute raw feature without any smoothing
            for bloop=1:n_bands,
                % Apply causal butterworth filter
                %bp_ieeg=butterfiltMK(ieeg,256,[bands(bloop,1) bands(bloop,2)],0,4);
                bp_ieeg=butterfilt4_causalEU(ieeg,256,[bands(bloop,1) bands(bloop,2)],0);
                %data=butterfilt4_causal(data,srate,flt,n_pad)
                
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
            
            if 0,
                % Code for checking results
                se_ftrs_z=zscore(se_ftrs')'; %z-score
                
                figure(1); clf();
                ax1=subplot(311);
                %imagesc(se_ftrs_z);
                h=plot(se_time_sec,se_ftrs_z);
                title(sprintf('%s-%s, Szr%d',soz_chans_bi{cloop,1},soz_chans_bi{cloop,2},sloop));
                
                ax2=subplot(312); plot(se_time_sec,se_class,'--b'); hold on;
                plot(se_time_sec,se_szr_class,'r-');
                axis tight;
                
                ax3=subplot(313);
                plot(time_dec,ieeg);
                axis tight;
                
                linkaxes([ax1 ax2 ax3],'x');
            end
            
            %% Grab only features during target window
            se_targ_ids=find(se_class>=0.5);
            se_ftrs=se_ftrs(:,se_targ_ids);
            se_time_sec=se_time_sec(se_targ_ids);
            se_class=se_class(se_targ_ids);
            se_szr_class=se_szr_class(se_targ_ids);
            
            n_tpt_ct(cloop)=n_tpt_ct(cloop)+length(se_targ_ids); % Keep track
            % of how many target observations were captured for each
            % electrode so that we can sample an equal number of non-target
            % examples
            
            %% Save just target window time points
            outdir=fullfile(root_dir,'EU_GENERAL','EU_GENERAL_FTRS','SE', ...
                num2str(sub_id));
            if ~exist(outdir,'dir'),
               mkdir(outdir); 
            end
            outfname=fullfile(outdir,sprintf('%d_%s_%s_szr%d',sub_id, ...
                soz_chans_bi{cloop,1},soz_chans_bi{cloop,2}, ...
                cli_szr_info(sloop).clinical_szr_num));
            szr_fname=cli_szr_info(sloop).clinical_fname;
            fprintf('Saving szr features to %s\n',outfname);
            save(outfname,'se_ftrs','se_time_sec','se_szr_class', ...
                'ftr_labels','szr_fname','targ_raw_ieeg','targ_raw_ieeg_sec');
            disp('there');
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

