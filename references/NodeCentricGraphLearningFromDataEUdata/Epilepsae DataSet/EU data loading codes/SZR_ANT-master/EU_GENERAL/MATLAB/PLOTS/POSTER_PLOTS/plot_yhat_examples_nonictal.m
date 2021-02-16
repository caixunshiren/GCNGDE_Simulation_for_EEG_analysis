%% This is the script used to plot nonictal examples for CLAE poster

%% Load yhat
if ismac,
    ftr_root='/Users/davidgroppe/PycharmProjects/SZR_ANT/EU_GENERAL/EU_GENERAL_FTRS/SE/';
    %yhat_root='/Users/davidgroppe/PycharmProjects/SZR_ANT/MODELS/genLogregSe_1_yhat';
else
    ftr_root='/home/dgroppe/EU_SE_FTRS/';
    yhat_root=sprintf('/home/dgroppe/EU_YHAT/');
end

example_szr=2;
switch example_szr,
    case 1,
        sub='1096';
        ftr_fname='/Users/davidgroppe/ONGOING/EU_SE_FTRS/1096_all/1096_HL3_HL4_109600102_0000.mat';
        load(ftr_fname);
        %         yhat_root='/Users/davidgroppe/PycharmProjects/SZR_ANT/MODELS/genLogregSe_yhat/';
        %        load(fullfile(yhat_root,'1096_HL3_HL4_phat_szr6.mat')); % Test Patient 1, Exmple Szr 1
        yhat_fname='/Users/davidgroppe/ONGOING/EU_YHAT/1096_genLogregSe_3/109600102_0000_yhat.mat';
        load(yhat_fname);
        yhat_chan_id=2;
        load('/Users/davidgroppe/ONGOING/EU_YHAT/1096_all_109600102_0000.mat')
        show_wind_sec=[-60 60]+890;
    otherwise, 
        sub=1125;
        chan1='HR11';
        chan2='HR12';
        ftr_fname=fullfile(ftr_root,sprintf('%d_all',sub), ...
            sprintf('%d_%s_%s_112500102_0000.mat',sub,chan1,chan2));
        load(ftr_fname);
        load('/home/dgroppe/GIT/SZR_ANT/temp_1125_inter.mat');
        %yhat_fname=fullfile(yhat_root,'1125_svmAesFinale_1/112500102_0000_yhat.npz');
        %yhat_fname='/Users/davidgroppe/ONGOING/EU_YHAT/1096_genLogregSe_3/109600102_0000_yhat.mat';
        %load(yhat_fname);
        load('/home/dgroppe/GIT/SZR_ANT/temp_1125_inter.mat');
        yhat_chan_id=1; % 1=HR11-HR12, 2=HR12-HR13
        show_wind_sec=[-60 60]+1173+60*15;
end

% This is hack!!! Need to redo this with proper trimmed mean and SD
for dloop=1:30,
    se_ftrs(dloop,:)=(se_ftrs(dloop,:)-mean(se_ftrs(dloop,:)))/ ...
        std(se_ftrs(dloop,:));
end

%%
% onset_tpt=min(find(se_szr_class>0.5));
% onset_sec=se_time_sec(onset_tpt);
fontsize=16;
prpl=[88, 29, 175]/255;
% t1=targ_raw_ieeg_sec(1);
% se_t1=se_time_sec(1);
switch example_szr,
    case 1,
        xtick=[0:20:120];
    otherwise,
        xtick=[0:20:120];
end

%show_wind_sec=[-60 60]+890;
start_eeg_tpt=findTpt(show_wind_sec(1),time_dec);
stop_eeg_tpt=findTpt(show_wind_sec(2),time_dec);

figure(1); clf();
set(gcf,'position',[85   230   844   452]);
ax1=axes('position',[0.1300    0.6793    0.7750    0.2457]);
plot(time_dec(start_eeg_tpt:stop_eeg_tpt)-show_wind_sec(1),ieeg(start_eeg_tpt:stop_eeg_tpt),'k-'); hold on;
axis tight;
set(gca,'xticklabels',[],'ytick',[],'xtick',xtick);
set(gca,'LineWidth',2);

% Plot Spectral Energy Features
start_se_tpt=findTpt(show_wind_sec(1),se_time_sec);
stop_se_tpt=findTpt(show_wind_sec(2),se_time_sec);

ax2=axes('position',[0.1300    0.365    0.7750    0.3157]);
imagesc(se_ftrs(:,start_se_tpt:stop_se_tpt)); hold on;
fprintf('Min max cbar values should be: %f %f\n',max(max(se_ftrs)),min(min(se_ftrs)));
ylim=get(gca,'ylim');
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

% Plot Classifier Output
ax3=axes('position',[0.1300    0.1100    0.7750    0.2535]);
%h=plot(se_time_sec,yhat,'b-');
h=area(se_time_sec(start_se_tpt:stop_se_tpt)-show_wind_sec(1), ...
    yhat_soz_chans(yhat_chan_id,start_se_tpt:stop_se_tpt));
set(h,'facecolor',prpl,'edgecolor',prpl);
ylim=get(gca,'ylim');
axis([[se_time_sec(start_se_tpt) se_time_sec(stop_se_tpt)]-show_wind_sec(1) 0 1]);
hold on;
%plot([1, 1]*onset_sec-se_t1,ylim,'r:','linewidth',3);
set(gca,'fontsize',fontsize,'ytick',[0:.25:1]);
%set(gca,'fontsize',fontsize,'ytick',[0:.25:1],'xtick',xtick);
set(gca,'LineWidth',2);
%'TickLength',[0.025 0.025]);
set(gca, 'Layer','top')



set(gcf,'paperpositionmode','auto');
print(1,'-djpeg',sprintf('yhat_nonszr_example_%d',example_szr));

disp('Done!');