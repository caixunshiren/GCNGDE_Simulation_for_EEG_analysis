% I think this is the script I used to plot SE features both with and
% without EDM for my posters. Not sure though. It's not running currently 
% (I must have changed paths or filenames). No sure how this differs from
% plot_edm_exampleV1.m

%% Load data
ftr_root='/Users/davidgroppe/PycharmProjects/SZR_ANT/EU_GENERAL/EU_GENERAL_FTRS/SE/';
sub='1096';
ftr_fname='1096_HL3_HL4_szr6.mat';
load(fullfile(ftr_root,sub,ftr_fname));


%%
show_tpts=[1:256*20];

onset_ids=find(se_szr_class>0.5);
onset_sec=se_time_sec(min(onset_ids));

ax1_pos=[0.1300    0.6338    0.7750    0.3412];

figure(1); clf();
set(gcf,'position',[131   268   971   424]);
% ax1=subplot(211);
ax1=axes('position',ax1_pos);
h=plot(targ_raw_ieeg_sec(show_tpts)-onset_sec,targ_raw_ieeg(show_tpts),'k-'); 
set(h,'linewidth',2);
hold on;
% axis tight;
%ylim=get(gca,'ylim');
ylim=[-750, 1300];
set(gca,'ylim',ylim,'xlim',[targ_raw_ieeg_sec(1) targ_raw_ieeg_sec(show_tpts(end))]-onset_sec);
%plot([1, 1]*onset_sec,ylim,'r-');
plot([0, 0],ylim,'r--','linewidth',3);
% title(ftr_fname);
ht=ylabel('Voltage');
set(ht,'fontsize',16);
set(gca,'ytick',[],'xtick',[]);

% ax2=subplot(312); 
% plot(se_time_sec,se_szr_class,'r-'); hold on;
% plot(se_time_sec,yhat,'b--')2
% axis tight;
ax2_pos=[0.1300    0.1    0.7750    0.533];
% ax2=subplot(212);
ax2=axes('position',ax2_pos);
ftr_tpts=find(se_time_sec<=targ_raw_ieeg_sec(show_tpts(end)));
imagesc(se_ftrs(:,ftr_tpts));
%xtick=get(gca,'xtick');
rel_se_stim_sec=se_time_sec-onset_sec;
xtick=[];
for tloop=-4:2:14,
    disp(tloop);
    xtick=[xtick findTpt(tloop,rel_se_stim_sec)];
end
set(gca,'fontsize',16);
set(gca,'xtick',xtick,'xticklabel',round(rel_se_stim_sec(xtick)),'ytick',[]);
h=xlabel('Seconds');
% h=plot(se_time_sec,se_ftrs);
% for floop=1:length(h),
%     clickText(h(floop),ftr_labels{floop},'none');
% end
% axis tight

% linkaxes([ax1 ax2 ax3],'x');
