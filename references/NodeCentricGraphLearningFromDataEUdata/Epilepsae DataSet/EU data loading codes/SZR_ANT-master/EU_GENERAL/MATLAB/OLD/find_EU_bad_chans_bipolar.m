%/Users/dgroppe/Desktop/25300102_0000.data 
%ts_data=pat.get_bin_signals();
%disp(size(ts_data));


%%
%pat=bin_file('/Users/dgroppe/Desktop/25300102_0000.data');
%pat=bin_file('/Volumes/ValianteLabEuData/EU/inv/pat_FR_1096/adm_1096102/rec_109600102/109600102_0000.data');
data_path='/Volumes/ValianteLabEuData/EU/inv/pat_FR_1096/adm_1096102/rec_109600102/';
bin_fname='109600102_0000.data';

data_path='/Volumes/ValianteLabEuData/EU/inv2/pat_FR_1073/adm_1073102/rec_107300102/';
bin_fname='107300102_0038.data';

pat=bin_file(fullfile(data_path,bin_fname));
disp(pat.a_samp_freq);
disp(pat.a_n_chan);
fprintf('# of samples=%d\n',(pat.a_n_samples));
%[data,time] = def_data_access(self, wsize, step, channs_cell, offset)
% wsize=pat.a_samp_freq;
% step=round(pat.a_samp_freq/10);
wsize=1/pat.a_samp_freq;
step=wsize;
channs_cell=pat.a_file_elec_cell;
offset=0;
[data,time] = pat.def_data_access(wsize, step, channs_cell, offset);
disp(size(data));


ignore_channels={'TP1','TP2','TP3','TP4','FP1','FP2','F3','F4','F7','F8','C1','C2','C3','C4', ...
    'P3','P4','O1','O2','T1','T2','T3','T4','T5','T6','FZ','CZ','PZ','EMG','ECG','EOG1','EOG2'};
ig_chans=intersect(ignore_channels,pat.a_file_elec_cell);
fprintf('Ignoring the following channels:\n');
disp(ig_chans);


%%
%/Users/dgroppe/Desktop/25300102_0000.data 
disp(pat.a_samp_freq);
disp(pat.a_n_chan);
disp(pat.a_file_elec_cell);
disp(pat.a_start_ts);
disp(pat.a_stop_ts);
fprintf('# of samples=%d\n',(pat.a_n_samples));
pat.a_channs_cell=pat.a_file_elec_cell;
pat.a_channs_cell=setdiff(pat.a_file_elec_cell,ignore_channels); % Remove ECG files
ieeg=pat.get_bin_signals([],[]);
disp(size(ieeg));



%% Downsample to 256 Hz
div_fact=pat.a_samp_freq/256;
% dec_fact=round(log2(div_fact));
n_tpt=round(pat.a_n_samples/div_fact);
n_chan=size(ieeg,1);
ieeg256=zeros(n_chan,n_tpt);
for c=1:n_chan,
    fprintf('Decimating chan %d/%d\n',c,n_chan);
    ieeg256(c,:)=decimate(ieeg(c,:),div_fact);
end
Fs=256;
clear ieeg

%% Do trimmed mean reference
ieeg256=ieeg256-repmat(trimmean(ieeg256,40),n_chan,1);


%% TODO whiten data
ieeg256=diff(ieeg256')';

%% Compute t-score of power spectrum from random 1 sec windows
mn_psd=zeros(1+Fs,n_chan);
sd_psd=mn_psd;
t_psd=mn_psd;

cfg=[];
cfg.Fs=Fs;
cfg.T=2;
cfg.K=4;
cfg.W=(cfg.K+1)/(2*cfg.T);
cfg.tapers=[cfg.T*cfg.W cfg.K];

n_rand_wind=1000;
for a=1:n_rand_wind,
    start_id=ceil(rand()*(n_tpt-2*Fs));
    use_ids=start_id:(start_id+2*Fs-1);
    [S,f]=mtspectrumc(diff(ieeg256(:,use_ids)'),cfg);
    %[S,f]=mtspectrumc(ieeg256(:,use_ids)',cfg);
    mn_psd=mn_psd+10*log10(S)/n_rand_wind;
end

%% Create ecog variable
global ecog
ecog=[];
ecog.filename=bin_fname;
ecog.psd.freqs=f;
ecog.psd.pwr=mn_psd';
ecog.ftrip.label=pat.a_channs_cell;
ecog.bad_chans=[];
ecog.lnnz=50;
bad_chan_GUI();

%% Save psd info
% psd_info=[];
% psd_info.psd=ecog.psd.pwr;
% psd_info.freqs=ecog.psd.freqs;
% psd_info.chan_labels=ecog.ftrip.label;
% psd_info.bad_chan_labels=ecog.bad_chans;
%good_chans=zeros(length(psd_info.chan_labels),1);
good_chans=zeros(length(ecog.ftrip.label),1);
good_chans(get_good_chans(ecog))=1;
% psd_info.good_chans=good_chans;
% psd_info.bin_fname=bin_fname;
dot_id=find(bin_fname=='.');
out_fname=[bin_fname(1:dot_id-1) '_psd.mat'];
fprintf('Saving psd info as %s\n',out_fname);

%%
save(out_fname,'good_chans','bin_fname','ecog');

%%
%[S,seg_times,f]=mtspecgramcDG(ecog.ftrip.trial{t}(a,:)',[cfg.T cfg.T/2],cfg);
cfg=[];
cfg.T=2;
cfg.K=4;
cfg.dB=1;
psd=mtaper_psd(ieeg256(1:4,use_ids),Fs,cfg);

%    T       : Duration of time window in seconds {required}
%    K       : Number of Slepian tapers (optional if W specified)
%    W       : ?? (optional if K specified)
%    segave  : If 1, the mean of the power spectrum across all segments is 
%              returned.  If 0, the power spectrum for each segment is
%              returned.
%    dB      : If 1, psd is returned in units of dB. If 0, psd is returned
%              in units of (uV^2)/Hz
%    rm_mean : If 'y', 'yes', or 1, the mean of channel for each trial will
%              be removed before the spectra are computed.
%    trim    : Throw out most extreme 100%*trim of data points.  For example
%              if trim=.1, 10% of the data will be ignored. {default: 0}

%% Plot interactive PSD and select bad channels

%% ave mean psd and data quality labels

%%

%% TODO
% stitch together contiguous files
% ADD SEIZURE CLASSES
% downsample to 256 Hz
% reduce dimensionality

%%
Fs=pat.a_samp_freq;
start_ts=pat.a_start_ts;
stop_ts=pat.a_stop_ts;
chan_names=pat.a_channs_cell;
out_fname='temp.mat';
save(out_fname,'ieeg','Fs','start_ts','stop_ts','chan_names');



%%
figure(1); clf;
plot(ts_data(:,[1:1000]+round(pat.a_n_samples/2))');