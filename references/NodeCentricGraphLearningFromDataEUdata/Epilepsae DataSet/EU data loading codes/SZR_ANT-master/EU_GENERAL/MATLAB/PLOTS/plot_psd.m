% Script for plotting pwr spectrum density produced by compute_psd_interictal.m
% Useful for identifying bad channels

sub=1096; %0.12=proportion of desired files actually sampled.
sub=1125; %0.12=proportion of desired files actually sampled.
sub=264; %0.14=proportion of desired files actually sampled.
sub=590; %0.04=proportion of desired files actually sampled.
sub=862; 


clear psd_samps
if ismac,
in_fname=sprintf('/Users/davidgroppe/PycharmProjects/SZR_ANT/EU_METADATA/PSD/%d_non_szr_psd.mat',sub);
else
   in_fname=sprintf('/home/dgroppe/GIT/SZR_ANT/EU_METADATA/PSD/%d_non_szr_psd.mat',sub);
end
load(in_fname);
[n_samp, n_freq, n_chan]=size(psd_samps);
good_psd_ids=find(psd_samps(:,1,1)); % # of files that were acutally computed
%(sometimes I quite PSD computation early

fprintf('%.2f=proportion of desired files actually sampled.\n',length(good_psd_ids)/n_samp);

mn_psd=squeeze(mean(psd_samps(good_psd_ids,:,:),1));
se_psd=squeeze(std(psd_samps(good_psd_ids,:,:),0,1))/sqrt(n_samp);

figure(1); clf();
subplot(1,2,1);
h=plot(f,mn_psd);
axis tight;
for cloop=1:length(h),
   clickText(h(cloop),[bipolar_labels{cloop,1} '-'  bipolar_labels{cloop,2}]);
end
xlabel('Hz');
ylabel('dB');
title('Mean PSD');

subplot(1,2,2);
hold on;
for cloop=1:n_chan,
    hh=shadedErrorBar(f,mn_psd(:,cloop),se_psd(:,cloop));
    set(hh.patch,'FaceColor',h(cloop).Color,'FaceAlpha',0.5);
    set(hh.mainLine,'Color',h(cloop).Color);
    clickText(hh.mainLine,[bipolar_labels{cloop,1} '-'  bipolar_labels{cloop,2}]);
    clickText(hh.patch,[bipolar_labels{cloop,1} '-'  bipolar_labels{cloop,2}]);
    for a=1:2,
        set(hh.edge(a),'Color',h(cloop).Color);
    end
end
axis tight;
title('Mean PSD with Stderr');
xlabel('Hz');