function [ieeg, time_dec, targ_raw_ieeg, targ_raw_ieeg_sec, targ_win_dec, szr_class_dec]=import_eu_clip(pat,chan1,chan2,targ_window,szr_class,Fs,sgramCfg)
% function [ieeg, time_dec, targ_raw_ieeg, targ_raw_ieeg_sec, targ_win_dec, szr_class_dec]=import_eu_clip(pat,chan1,chan2,targ_window,szr_class,Fs,sgramCfg);
%
% Inputs:
%  pat - Epilepsiae data object 
%  chan1 - String of monopolar channel name
%  chan2 - String of bipolar channel name
%  targ_window - 
%  szr_class - 0=no szr, 1=clinical szr, -1=subclinical szr
%  Fs - sampling rate
%  sgramCfg - spectrogram configuration file (used for figuring out
%             sgramCfg.start_time)
%
% Outputs:
%  ieeg - bipolar data from entire clip downsampled to 256 Hz
%  time_dec - time (in seconds) relative to ieeg
%  targ_raw_ieeg - raw eeg of seizure (nan if no szrs in clip)
%  targ_raw_ieeg_sec - time (in sec) relative to targ_raw_ieeg
%  targ_win_dec - stim class (1=stim,0=no stim) downsampled to 256 Hz
%  szr_class_dec - szr_class (1=stim,0=no stim) downsampled to 256 Hz

% Import entire clip (typically 1 hour long)
%             ieeg_labels=cell(n_chan,1);
%pat.a_channs_cell={soz_chans_bi{cloop,1}}; % Channel to import
pat.a_channs_cell={chan1};
ieeg_temp1=pat.get_bin_signals(1,pat.a_n_samples);

%pat.a_channs_cell={soz_chans_bi{cloop,2}}; % Channel to import
pat.a_channs_cell={chan2};
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
n_ieeg_dec_tpts=length(ieeg);
clear ieeg_time_sec_pre_decimate;

% Clip raw szr data (if present)
targ_raw_ieeg_tpts=(targ_win_dec>0);
if sum(targ_raw_ieeg_tpts)>0,
    %targ_raw_ieeg=ieeg(targ_raw_ieeg_tpts);
    %             targ_raw_ieeg_sec=time_dec(targ_raw_ieeg_tpts);
    
    % Compute start and stop time points for seizure for spectrogram
    % Extend target window forward and backward a bit to capture
    % full time window
    start_targ_id=min(find(targ_raw_ieeg_tpts))-sgramCfg.T*sgramCfg.Fs;
    stop_targ_id=max(find(targ_raw_ieeg_tpts))+sgramCfg.T*sgramCfg.Fs;
    if start_targ_id<1,
        start_targ_id=1;
    end
    if stop_targ_id>n_ieeg_dec_tpts,
        stop_targ_id=n_ieeg_dec_tpts;
    end
    sgramCfg.start_time=time_dec(start_targ_id);
    targ_raw_ieeg=ieeg(start_targ_id:stop_targ_id);
    targ_raw_ieeg_sec=time_dec(start_targ_id:stop_targ_id);
else
    targ_raw_ieeg=nan;
    targ_raw_ieeg_sec=nan;
end