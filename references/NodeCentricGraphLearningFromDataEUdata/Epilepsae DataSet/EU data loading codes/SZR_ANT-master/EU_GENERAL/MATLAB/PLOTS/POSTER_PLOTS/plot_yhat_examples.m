% This script makes plots of example szrs and classifier outputs that I
% used for posters. It shows the raw EEG, SE features, classifier output
% and clinician onset. 
%
% You need to manually enter which szr and channel you want to show
%
% apply_saved_model_to_szr.py needs to be run on that subject with the
% desired classifier for this to work

save_em=0;


%% Load yhat
%load('/Users/davidgroppe/PycharmProjects/SZR_ANT/MODELS/genLogregSe_yhat/1096_HL1_HL2_phat_szr6.mat')
if ismac,
    ftr_root='/Users/davidgroppe/PycharmProjects/SZR_ANT/EU_GENERAL/EU_GENERAL_FTRS/SE/';
    yhat_root='/Users/davidgroppe/PycharmProjects/SZR_ANT/MODELS/genLogregSe_1_yhat';
else
    ftr_root='/home/dgroppe/GIT/SZR_ANT/EU_GENERAL/EU_GENERAL_FTRS/SE/';
    yhat_root='/home/dgroppe/GIT/SZR_ANT/MODELS/svmAesFinale_1_yhat/';
end

example_szr=2;
sec_cutoff=30; %show this much of the seizure clip
% NOTE szr_num indices start at 0
switch example_szr,
    case 1,
        sub='1096';
        ftr_fname='1096_HL3_HL4_szr6.mat';
        load(fullfile(ftr_root,sub,ftr_fname)); % includes se_ftrs
        %yhat_root='/Users/davidgroppe/PycharmProjects/SZR_ANT/MODELS/genLogregSe_yhat/';
        load(fullfile(yhat_root,'1096_HL3_HL4_phat_szr6.mat')); % Test Patient 1, Exmple Szr 1
    case 2,
        %Example of a seizure with late stim latency
        sub=1125; % Test patient 4
        szr_num=7;
        chan1='HR12';
        chan2='HR13';
%        ftr_fname=sprintf('1125_HR11_HR12_szr%d.mat',szr_num);
        %yhat_root='/Users/davidgroppe/Desktop/genLogregSe_1_yhat/';
        %yhat_root='/Users/davidgroppe/PycharmProjects/SZR_ANT/MODELS/genLogregSe_yhat/';
        %load(fullfile(yhat_root,sprintf('1125_HR11_HR12_phat_szr%d.mat',szr_num)));
    case 3,
        % Example of a seizure with good stim latency
        sub=264; % Test patient 1
        szr_num=6;
        chan1='TRA1';
        chan2='TRA2';
    case 4,
        % Example of a missed szr
        sub=862; % Test patient 3
        szr_num=0;
%         chan1='IHB2';
%         chan2='IHB3';
        chan1='IHB3';
        chan2='IHB4';
end
ftr_fname=sprintf('%d_%s_%s_szr%d.mat',sub,chan1,chan2,szr_num);
fprintf('Loading %s\n',ftr_fname);
load(fullfile(ftr_root,num2str(sub),ftr_fname));
yhat_fname=sprintf('%d_%s_%s_phat_szr%d.mat',sub,chan1,chan2,szr_num);
fprintf('Loading %s\n',yhat_fname);
load(fullfile(yhat_root,yhat_fname));

%%
onset_tpt=min(find(se_szr_class>0.5));
onset_sec=se_time_sec(onset_tpt);
fontsize=16;
prpl=[88, 29, 175]/255;
t1=targ_raw_ieeg_sec(1);
se_t1=se_time_sec(1);
switch example_szr,
    case 1,
        xtick=[0:20:120];
    otherwise,
        xtick=[0:10:120];
end

figure(1); clf();
set(gcf,'position',[85   230   844   452]);

% PLOT EEG TRACE
stop_tpt=findTpt(sec_cutoff,targ_raw_ieeg_sec-onset_sec);
ax1=axes('position',[0.1300    0.6793    0.7750    0.2457]);
% ax1=subplot(311);
plot(targ_raw_ieeg_sec(1:stop_tpt)-onset_sec,targ_raw_ieeg(1:stop_tpt),'k-'); hold on;
axis tight;
ylim=get(gca,'ylim');
%plot([1, 1]*onset_sec-t1,ylim,'r:','linewidth',3);
plot([0, 0],ylim,'r:','linewidth',3);
set(gca,'xticklabels',[],'ytick',[],'xtick',xtick);
set(gca,'LineWidth',2);
set(gca,'xlim',[se_time_sec(1)-onset_sec sec_cutoff]);

% PLOT NORMALIZED FEATURES
ax2=axes('position',[0.1300    0.365    0.7750    0.3157]);
stop_tpt=findTpt(sec_cutoff,se_time_sec-onset_sec);
%imagesc(se_ftrs); hold on; %raw features
use_se_z=ftrs_z(:,1:stop_tpt);
imagesc(use_se_z); hold on;
fprintf('Min max cbar values should be: %f %f\n',max(max(use_se_z)),min(min(use_se_z)));
ylim=get(gca,'ylim');
plot([1, 1]*onset_tpt,ylim,'r:','linewidth',3);
set(gca,'xtick',[],'ytick',[]);
axis tight

% Add colorbar
%ax2=axes('position',[0.1300    0.365    0.7750    0.3157]);
pos=[.91 .365 .02 .3157];
%pos=[.09 .365 .02 .3157];
limits=[-1, 1];
cmapName='parula';
units='';
nTick=0;
fontSize=12;
unitLocation='top';
hCbar = cbarDGplus(pos,limits,cmapName,nTick,units,unitLocation,fontSize);

% SMOOTH CLASSIFIER PREDICTIONS
yhat_smooth=zeros(size(yhat));
for a=1:length(yhat),
    back_tpt=a-20+1; % causal moving window is about 2 sec, 20 time points back
    if back_tpt<1,
        back_tpt=1;
    end
    yhat_smooth(a)=mean(yhat(back_tpt:a));
end

% PLOT CLASSIFIER PREDICTIONS
%stop_tpt=findTpt(sec_cutoff,se_time_sec-onset_sec);
% ax3=subplot(313); 
ax3=axes('position',[0.1300    0.1100    0.7750    0.2535]);
%h=plot(se_time_sec,yhat,'b-');
h=area(se_time_sec(1:stop_tpt)-onset_sec,yhat_smooth(1:stop_tpt));
set(h,'facecolor',prpl,'edgecolor',prpl);
axis([[se_time_sec(1) se_time_sec(end)]-onset_sec 0 1]);
hold on;
set(gca,'xlim',[se_time_sec(1)-onset_sec, sec_cutoff]);
ylim=get(gca,'ylim');
%plot([1, 1]*onset_sec-se_t1,ylim,'r:','linewidth',3);
plot([0, 0],ylim,'r:','linewidth',3);
set(gca,'fontsize',fontsize,'ytick',[0:.25:1],'xtick',xtick);
set(gca,'LineWidth',2);
%'TickLength',[0.025 0.025]);
set(gca, 'Layer','top')

if save_em,
out_fig_fname=sprintf('yhat_example_%d',example_szr);
fprintf('Exporting fig 1 to %s\n',out_fig_fname);
set(gcf,'paperpositionmode','auto');
print(gcf,'-djpeg',out_fig_fname);
else
   fprintf('NOT saving plot to disk.\n'); 
end

disp('Done!');