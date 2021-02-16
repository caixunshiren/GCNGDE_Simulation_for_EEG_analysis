% This script computes the power spectral density from interictal data in
% 20 randomly sampled files. It is used for identifying bad channels and
% normalizing spectrograms.
%
% Output is saved to SZR_ANT/EU_METADATA/PSD

%sub_id=1096; % PARTIALLY DONE
%sub_id=620; % patially done
%sub_id=264; % DONE
% sub_id=590; %
% sub_id=862;%
% sub_id=565; %
sub_id=253; %
%sub_id=273; %
%sub_id=1125; % DONE
%sub_id=1077; %
%do_subs=[620, 590, 862, 565, 253, 273, 1077];
%do_subs=[273, 958, 565]; % DONE
% need to do
do_subs=[922, 970];
%do_subs=970;
do_subs=[115];
do_subs=[442];
do_subs=1096;

for sub_id=do_subs,
    
    if ismac,
        root_dir='/Users/davidgroppe/PycharmProjects/SZR_ANT/';
    else
        root_dir='/home/dgroppe/GIT/SZR_ANT/';
    end
    
    
    %% Derive list of bipolar channels
    bipolar_labels=derive_bipolar_pairs(sub_id);
    n_chan=size(bipolar_labels,1);
    fprintf('# of Channels: %d\n',n_chan);
    
    
    %% Get list of all files and the timing of all szrs (clinical and subclinical)
    file_info=get_fnames_and_szr_times(sub_id);
    n_files=length(file_info);
    
    %% SGRAM Params
    psd_params=[];
    psd_params.Fs=256;
    psd_params.T=2; %1 second window
    psd_params.K=3; % # of tapers
    psd_params.fpass=[0 .4*psd_params.Fs];
    W=(psd_params.K+1)/(2*psd_params.T);
    psd_params.tapers=[W*psd_params.T psd_params.K]; %[TW K]
    psd_params.rm_mean=0;
    psd_params.trialave=0;
    
    
    %% Loop over SOZ electrodes
    n_rand_files=20; % TODO ?? increase this
    temp_perm=randperm(n_files);
    rand_files_used=temp_perm(1:n_rand_files);
    for floop=1:n_rand_files,
        % Read header
        full_data_fname=get_data_fullfname(sub_id,file_info(floop).fname); % TODO ?? make this a random selection
        fprintf('Loading file %d/%d (%d available) %s\n',floop,n_rand_files, ...
            n_files,file_info(floop).fname);
        pat=bin_file(full_data_fname);
        Fs=pat.a_samp_freq;
        if isempty(Fs),
            error('Could not find file: %s',full_data_fname);
        end
        fprintf('Fs=%f\n',Fs);
        %fprintf('# of monopolar chans %d\n',pat.a_n_chan);
        fprintf('# of samples=%d\n',(pat.a_n_samples));
        
        for cloop=1:n_chan,
            % If first channel for this clip, compute ictal labels
            fprintf('Channel %d/%d\n',cloop,n_chan);
            fprintf('Working on chan %s-%s\n',bipolar_labels{cloop,1},bipolar_labels{cloop,1});
            % Import entire clip (typically 1 hour long)
            pat.a_channs_cell={bipolar_labels{cloop,1}}; % Channel to import
            ieeg_temp1=pat.get_bin_signals(1,pat.a_n_samples);
            
            pat.a_channs_cell={bipolar_labels{cloop,2}}; % Channel to import
            ieeg_temp2=pat.get_bin_signals(1,pat.a_n_samples);
            
            ieeg=ieeg_temp1-ieeg_temp2;
            ieeg_time_sec_pre_decimate=[0:(length(ieeg)-1)]/Fs; % time relative to start of file
            clear ieeg_temp1 ieeg_temp2;
            
            if cloop==1,
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
            end
                
                %% Downsample if necessary
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
            
            % Find out how many non-szr time points there are in the file
            
            %% Randomly sample # of time pts * .00015 1 sec windows and compute PSD
            if cloop==1,
                % Randomly select time points to sample
                n_non_szr_tpts=sum(targ_win_dec==0);
                n_fft_wind=round(n_non_szr_tpts*.00015);
                wind_len=psd_params.T*psd_params.Fs;
                rand_tpts=randi(length(ieeg)-wind_len,n_fft_wind);
            end
            use_fft_ids=zeros(n_fft_wind,1);
            ieeg_clips=zeros(wind_len,n_fft_wind);
            for fft_loop=1:n_fft_wind,
                temp_tpt_ids=rand_tpts(fft_loop):rand_tpts(fft_loop)+psd_params.T*psd_params.Fs-1;
                temp_szr_tpts=sum(targ_win_dec(temp_tpt_ids));
                if temp_szr_tpts==0,
                    ieeg_clips(:,fft_loop)=ieeg(temp_tpt_ids);
                    use_fft_ids(fft_loop)=1;
                end
            end
            
            % Compute PSD
            [S, f]=mtspectrumc(ieeg_clips,psd_params);
            if floop==1 && cloop==1,
                % preallocate memory
                psd_samps=zeros(n_files,length(f),n_chan);
            end
            psd_samps(floop,:,cloop)=mean(10*log10(S(:,use_fft_ids>0)),2);
           
            
        end
        % Save results to disk
        outdir=fullfile(root_dir,'EU_METADATA','PSD');
        if ~exist(outdir,'dir'),
            mkdir(outdir);
        end
        outfname=fullfile(outdir,sprintf('%d_non_szr_psd',sub_id));
        fprintf('Saving non-szr PSD to %s\n',outfname);
        save(outfname,'psd_samps','f','bipolar_labels', ...
            'file_info','rand_files_used');
        
    end
end

disp('Done!!');

