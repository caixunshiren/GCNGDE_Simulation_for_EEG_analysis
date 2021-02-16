% This script loads the power spectra density from randomly sampled
% non-ictal data and plots it with a GUI that allows one to select bad
% channels.
%
% Bad channels are then output as a textfile here:
%/Users/davidgroppe/PycharmProjects/SZR_ANT/EU_METADATA/BAD_CHANS

%% Load PSD data
%sub=264; %20 (0.14)=# (proportion) of total files actually sampled.
%sub=565; %20 (0.09)=# (proportion) of total files actually sampled.
% sub=1096; % 20 (0.12)=# (proportion) of total files actually sampled.
%sub=1125; %20 (0.12)=# (proportion) of total files actually sampled.
%sub=273; %20 (0.10)=# (proportion) of total files actually sampled.
% sub=1077; %20 (0.11)=# (proportion) of total files actually sampled.
sub=862; %20 (0.09)=# (proportion) of total files actually sampled.
% sub=253;%20 (0.07)=# (proportion) of total files actually sampled.
% sub=590; %20 (0.08)=# (proportion) of total files actually sampled.
% sub=620; %20 (0.08)=# (proportion) of total files actually sampled.
% sub=958; %20 (0.09)=# (proportion) of total files actually sampled.
% sub=922; %20 (0.18)=# (proportion) of total files actually sampled.
% sub=970; %20 (0.10)=# (proportion) of total files actually sampled.
% sub=115; %20 (0.08)=# (proportion) of total files actually sampled.
% sub=442; %20 (0.11)=# (proportion) of total files actually sampled.

if ismac,
psd_fname=sprintf('/Users/davidgroppe/PycharmProjects/SZR_ANT/EU_METADATA/PSD/%d_non_szr_psd.mat',sub);
else
   psd_fname=sprintf('/home/dgroppe/GIT/SZR_ANT/EU_METADATA/PSD/%d_non_szr_psd.mat',sub);
end
load(psd_fname);
good_psd_ids=find(psd_samps(:,1,end)); % # of files that were acutally computed
%(sometimes I quite PSD computation early
n_samp=size(psd_samps,1);
fprintf('Sub %d\n',sub);
fprintf('%d (%.2f)=# (proportion) of total files actually sampled.\n', ...
    length(good_psd_ids),length(good_psd_ids)/n_samp);


%%
n_chan=size(bipolar_labels,1);
n_files=size(psd_samps,1);
chan_labels_str=cell(n_chan,1);
for a=1:n_chan,
   chan_labels_str{a}=[bipolar_labels{a,1} '-'  bipolar_labels{a,2}];
end


%% Create ecog variable
good_file_ids=find(psd_samps(:,1,1)>0);
fprintf('%d files sampled from\n',length(good_file_ids));
mn_psd=squeeze(mean(psd_samps(good_file_ids,:,:),1));
se_psd=squeeze(std(psd_samps(good_file_ids,:,:),0,1))/sqrt(n_files);
global ecog
ecog=[];
ecog.filename=psd_fname;
ecog.psd.freqs=f;
ecog.psd.pwr=mn_psd';
%ecog.ftrip.label=pat.a_channs_cell;
ecog.ftrip.label=chan_labels_str;
ecog.bad_chans=[];
ecog.lnnz=50;
bad_chan_GUI();


disp('Select bad channels with GUI and then run rest of script manually.');
return



%% Plot good & bad chams with stderr
good_chan_ids=get_good_chans(ecog);
bad_chan_ids=setdiff(1:n_chan,good_chan_ids);
figure(10);
clf();
subplot(1,2,1); hold on;
hh=plot(f,mn_psd(:,good_chan_ids));
ct=0;

for a=good_chan_ids,
    ct=ct+1;
    h=shadedErrorBar(f,mn_psd(:,a),se_psd(:,a),[]);
    h.patch.FaceAlpha=0.2;
    h.patch.FaceColor=hh(ct).Color;
    h.mainLine.Color=hh(ct).Color;
    clickText(h.mainLine,chan_labels_str{a});
    %shadedErrorBar(x,mean(y,1),std(y),'g');
    %plot(f,mn_psd(:,good_chan_ids));
end
axis tight;
xlabel('Hz');
ylabel('Log Pwr');
title('Good Channels');

subplot(1,2,2); hold on;
hh=plot(f,mn_psd(:,bad_chan_ids));
ct=0;
for a=bad_chan_ids,
    ct=ct+1;
    h=shadedErrorBar(f,mn_psd(:,a),se_psd(:,a),[]);
    h.patch.FaceAlpha=0.2;
    h.patch.FaceColor=hh(ct).Color;
    h.mainLine.Color=hh(ct).Color;
    clickText(h.mainLine,chan_labels_str{a});
    %shadedErrorBar(x,mean(y,1),std(y),'g');
    %plot(f,mn_psd(:,good_chan_ids));
end
axis tight;
xlabel('Hz');
title('Bad Channels');
ylabel('Log Pwr');


%% Output bad channels to text file
out_fname=sprintf('/Users/davidgroppe/PycharmProjects/SZR_ANT/EU_METADATA/BAD_CHANS/bad_chans_%d.txt',sub);
fprintf('Saving bad chans to %s\n',out_fname)
fid=fopen(out_fname,'w');
if length(bad_chan_ids)>0
    for a=bad_chan_ids,
        fprintf(fid,'%s\n',chan_labels_str{a});
    end
else
    fprintf(fid,'None\n');
end
fclose(fid);