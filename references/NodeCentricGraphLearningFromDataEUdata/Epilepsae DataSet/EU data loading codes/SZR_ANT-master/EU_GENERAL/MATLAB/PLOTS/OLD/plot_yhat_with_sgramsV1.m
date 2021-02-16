% This script plots raw data, true class, class hat, features (z-scored),
% spectrogram, and t-scored spectrogram for a szr
%
% It is useful for seeing what features pickup and miss

%% Load raw data and spectrogram of raw data
% sub=1096;
% chan='HL2-HL3';
% % chan='HL3-HL4';
% szr_num=6;

sub=565;
chan='HL4-HL5';
szr_num=6;

model_name='genLogregSe_1_yhat';

if ismac,
    szr_ant_root='/Users/davidgroppe/PycharmProjects/SZR_ANT/';
else
   szr_ant_root='/home/dgroppe/GIT/SZR_ANT/';
end
ftr_root=fullfile( szr_ant_root,'EU_GENERAL/EU_GENERAL_FTRS/SE/');
yhat_root=fullfile(szr_ant_root,'MODELS',model_name);

dash_id=find(chan=='-');
chan1=chan(1:dash_id-1);
chan2=chan(dash_id+1:end);
ftr_fname_loaded=sprintf('%d_%s_%s_szr%d.mat',sub,chan1,chan2,szr_num);
%fprintf('Loading %s\n',ftr_fname_loaded);
%ftr_fname='1096_HL3_HL4_szr6.mat';
%load(fullfile(ftr_root,num2str(sub),ftr_fname_loaded));
in_fname_ftrs=fullfile(ftr_root,num2str(sub),ftr_fname_loaded);
fprintf('Loading %s\n',in_fname_ftrs);
load(in_fname_ftrs);

% Remove _ from feature labels
for a=1:size(ftr_labels,1),
    use_ids=find(ftr_labels{a}~='_');
    ftr_labels{a}=ftr_labels{a}(use_ids);
end

% Scale all features from 0 to 1
% for a=1:size(ftr_labels,1),
%     se_ftrs(a,:)=se_ftrs(a,:)-min(se_ftrs(a,:));
%     se_ftrs(a,:)=se_ftrs(a,:)/max(se_ftrs(a,:));
% end

%% Load yhat and z-scored features (ftrs_z)
%load('/Users/davidgroppe/PycharmProjects/SZR_ANT/MODELS/genLogregSe_yhat/1096_HL1_HL2_phat_szr6.mat')
%yhat_root='/Users/davidgroppe/PycharmProjects/SZR_ANT/MODELS/genLogregSe_3_yhat/';
%load(fullfile(yhat_root,'1096_HL3_HL4_phat_szr6.mat'))
%load(fullfile(yhat_root,sprintf('%d_%s_%s_phat_szr%d.mat',sub,chan1,chan2,szr_num)));
in_fname_yhat=fullfile(yhat_root,sprintf('%d_%s_%s_phat_szr%d.mat',sub,chan1,chan2,szr_num));
fprintf('Loading %s\n',in_fname_yhat);
load(in_fname_yhat);


% Scale all features from 0 to 1
% for a=1:size(ftr_labels,1),
%     ftrs_z(a,:)=ftrs_z(a,:)-min(ftrs_z(a,:));
%     ftrs_z(a,:)=ftrs_z(a,:)/max(ftrs_z(a,:));
% end


%% Load PSD
% /Users/davidgroppe/PycharmProjects/SZR_ANT/EU_METADATA/PSD/1096_non_szr_psd.mat
psd_fname=sprintf('%s/EU_METADATA/PSD/%d_non_szr_psd.mat',szr_ant_root,sub);
if exist(psd_fname,'file'),
    psd_exists=1;
    fprintf('Loading %s\n',psd_fname);
    load(psd_fname);
    chan_id=NaN;
    for a=1:size(bipolar_labels,1),
        if strcmpi(bipolar_labels{a,1},chan1) && strcmpi(bipolar_labels{a,2},chan2),
            chan_id=a;
        end
    end
    if isnan(chan_id),
        error('Could not find index to channel %s-%s in PSD data',chan1,chan2);
    end
    good_psd_ids=find(psd_samps(:,1,chan_id));
    mn_psd=squeeze(mean(psd_samps(good_psd_ids,:,:),1));
    se_psd=squeeze(std(psd_samps(good_psd_ids,:,:),1,1))/sqrt(size(psd_samps,1));
else
    psd_exists=0;
    mn_psd=zeros(length(sgram_f),1);
    se_psd=ones(length(sgram_f),1);
end


%% Plot Mean PSD
if psd_exists,
    used_file_ids=find(psd_samps(:,1,chan_id)>0);
    figure(2); clf; hold on;
    %plot(sgram_f,mn_psd(:,chan_id),'k-','linewidth',2);
    h=shadedErrorBar(sgram_f,mn_psd(:,chan_id),se_psd(:,chan_id),[]);
    plot(sgram_f,squeeze(psd_samps(used_file_ids,:,chan_id)));
    hold on;
end
xlabel('Hz');
ylabel('dB');
title('Non-ictal PSD for this chan (lines are diff clips)');
axis tight;


%% Bands
bands=[0 4; 4 8; 8 13; 13 30; 30 50; 70 100];
band_labels={'Delta','Theta','Alpha','Beta','Gamma','HGamma'};
n_band=size(bands,1);
band_rgb=zeros(n_band,3);
band_rgb(1,:)=[250 175 63]/255; % Delta (Orange)
band_rgb(2,:)=[1 0 0]; % Theta (red)
band_rgb(3,:)=[158 40 226]/255; % Alpha (purple)
band_rgb(4,:)=[0 0 1]; % Beta (blue)
band_rgb(5,:)=[0 1 0]; % Green (gamma)
band_rgb(6,:)=[1 0 1]; % High gamm (magenta)

%% Plot raw data, yhat, ftrs, & spectrograms
figure(1); clf();
set(gcf,'position',[85   100   844   582]);

% Raw data
ax1=subplot(611);
plot(targ_raw_ieeg_sec,targ_raw_ieeg,'b-');
axis tight;
ht=title(ftr_fname_loaded);
set(ht,'interpreter','none');
ylabel('Volatge');

% True Szr class and estimated szr class (note szrs start at 5 sec before
% clinician onset
ax2=subplot(612); hold on;
area(se_time_sec,yhat);
plot(se_time_sec,se_szr_class,'r-','linewidth',2);
%plot(se_time_sec,yhat,'b--');
axis tight;
ylabel('P(Szr)');
% set(gca,'ylim',[0 1],'ytick',[0, 1]);

% Z-scored ftrs
ax3=subplot(613);
h=plot(se_time_sec,ftrs_z); hold on;
for floop=1:length(h),
    clickText(h(floop),ftr_labels{floop});
end
h=plot(se_time_sec,ftrs_z(1,:),'-','linewidth',2,'color',[250 175 63]/256); % Delta
clickText(h,'Delta');
h=plot(se_time_sec,ftrs_z(2,:),'r-','linewidth',2); % Theta
clickText(h,'Theta');
h=plot(se_time_sec,ftrs_z(3,:),'-','linewidth',2,'color',[158 40 226]/256); % Alpha
clickText(h,'Alpha');
h=plot(se_time_sec,ftrs_z(4,:),'b-','linewidth',2); % Beta
clickText(h,'Beta');
h=plot(se_time_sec,ftrs_z(5,:),'g-','linewidth',2); % Gamma
clickText(h,'Gamma');
h=plot(se_time_sec,ftrs_z(6,:),'m-','linewidth',2); % High Gamma
clickText(h,'HGamma');
axis tight
ylabel('hilbert se(z)');

% Spectrogram of Szr with freq bands marked with white lines
ax5=subplot(615);
h=imagesc(sgram_S');
hold on;
ytick=get(gca,'ytick');
yticklabels=cell(length(ytick),1);
for a=1:length(ytick),
    yticklabels{a}=num2str(sgram_f(ytick(a)));
end
xtick=get(gca,'xtick');
xticklabels=cell(length(xtick),1);
for a=1:length(xtick),
    xticklabels{a}=num2str(round(sgram_t(xtick(a))));
end
set(gca,'ydir','normal','xticklabel',xticklabels,'yticklabel',yticklabels);
ylabel('Hz');
axis tight
% Indicate borders between frequency bands with white lines
xlim=get(gca,'xlim');
for bloop=1:n_band,
    hz_id=findTpt(bands(bloop,2),sgram_f);
%     disp(hz_id);
    plot(xlim,[1 1]*hz_id,'w-');
end
ht=title('Raw Sgram (dB)');

if psd_exists,
    sgram_band=zeros(n_band,length(sgram_t));
    % t-scored Spectrogram of Szr with freq bands marked with white lines
    ax6=subplot(616);
    sgram_tscore=sgram_S-repmat(mn_psd(:,chan_id)',length(sgram_t),1);
    sgram_tscore=sgram_tscore./repmat(se_psd(:,chan_id)',length(sgram_t),1);
    h=imagesc(sgram_tscore'); hold on;
    ytick=get(gca,'ytick');
    yticklabels=cell(length(ytick),1);
    for a=1:length(ytick),
        yticklabels{a}=num2str(sgram_f(ytick(a)));
    end
    xtick=get(gca,'xtick');
    xticklabels=cell(length(xtick),1);
    for a=1:length(xtick),
        xticklabels{a}=num2str(round(sgram_t(xtick(a))));
    end
    set(gca,'ydir','normal','xticklabel',xticklabels,'yticklabel',yticklabels);
    ylabel('Hz');
    xlabel('Sec');
    axis tight
    % Indicate borders between frequency bands with white lines
    xlim=get(gca,'xlim');
    for bloop=1:n_band,
        hz_id=findTpt(bands(bloop,2),sgram_f);
        plot(xlim,[1 1]*hz_id,'w-');
        
        hz_id_low=findTpt(bands(bloop,1),sgram_f);
        fprintf('Band %d Hz ids: %d to %d\n',bloop,hz_id_low,hz_id);
        sgram_band(bloop,:)=mean(sgram_tscore(:,hz_id_low:hz_id),2);
    end
    ht=title('Sgram (t)');
end

% t-scored sgram ftrs
ax4=subplot(614); hold on;
for bloop=1:n_band,
    h=plot(sgram_t,sgram_band(bloop,:),'-','linewidth',2,'color', ...
        band_rgb(bloop,:));
    clickText(h,band_labels{bloop});
end
axis tight
legend(band_labels,'location','northeast');
ylabel('sgram pwr(t)');

linkaxes([ax1 ax2 ax3 ax4],'x');
% print(1,'-djpeg','