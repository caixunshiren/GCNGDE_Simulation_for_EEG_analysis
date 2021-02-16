% Script used to plot examples of feature extraction in CLAE poster

%% Load data
if ismac,
    ftr_root='/Users/davidgroppe/PycharmProjects/SZR_ANT/EU_GENERAL/EU_GENERAL_FTRS/SE/';
else
   ftr_root='/home/dgroppe/GIT/SZR_ANT/EU_GENERAL/EU_GENERAL_FTRS/SE/'; 
end
sub='1096';
fontsize=16;
print_em=1;

%% Load features
ftr_fname='1096_HL3_HL4_szr6.mat';
load(fullfile(ftr_root,sub,ftr_fname));


%% Load subsampled data that I can normalize szr data
nrm_ftr_fname='1096_HL3_HL4_subsamp.mat';
load(fullfile(ftr_root,sub,nrm_ftr_fname),'subsamp_se_ftrs');
[trimMn, trimSd]=trimMeanAndSd(subsamp_se_ftrs',50);

n_dim=size(se_ftrs,1);
for a=1:n_dim,
    se_ftrs(a,:)=(se_ftrs(a,:)-trimMn(a))/trimSd(a);
end



%% Plot raw spectral power with onset FIGURE 2
show_tpts=[1:256*20];
onset_ids=find(se_szr_class>0.5);
onset_sec=se_time_sec(min(onset_ids));


figure(2); clf();
set(gcf,'position',[131   416   868   276]);
ftr_tpts=find(se_time_sec<=targ_raw_ieeg_sec(show_tpts(end)));
show_dat=se_ftrs(1:6,ftr_tpts);
full_show_dat=se_ftrs(:,ftr_tpts);
% Get max and min values from EDM smoothed features (Fig 3) so that Fig 2
% and Fig 3 can share the same colorscale
mx_cmap=max(max(full_show_dat));
min_cmap=min(min(full_show_dat));
imagesc(show_dat,[min_cmap, mx_cmap]);

rel_se_stim_sec=se_time_sec-onset_sec;
xtick=[];
xticklabel=[];
ct=0;
clear onset_x;
for tloop=-4:2:14,
    disp(tloop);
    xtick=[xtick findTpt(tloop,rel_se_stim_sec)];
    if tloop==0,
        onset_x=xtick(end);
    end
    ct=ct+1;
    xticklabel{ct}=num2str(tloop);
end
set(gca,'xtick',xtick,'xticklabel',xticklabel,'ytick',[]);
ylim=get(gca,'ylim');
set(gca,'fontsize',fontsize);
hold on;
plot([1, 1]*onset_x,ylim,'r:','linewidth',3);

show_time_wind=rel_se_stim_sec(ftr_tpts);


if print_em,
    fprintf('Saving figure to edm_example_raw_nrg\n');
    set(gcf,'paperpositionmode','auto');
    print(2,'-djpeg','edm_example_raw_nrg');
else
    fprintf('Not saving figures!\n');
end

% Add colorbar
% pos=[.91 .43 .02 .533];
% limits=[-1, 1];
% cmapName='parula';
% units='';
% nTick=0;
% fontSize=12;
% unitLocation='top';
% hCbar = cbarDGplus(pos,limits,cmapName,nTick,units,unitLocation,fontSize);


%% Plot spectral energy with EDM smoothing FIGURE 3
figure(3); clf();
set(gcf,'position',[131   416   868   276]);
ftr_tpts=find(se_time_sec<=targ_raw_ieeg_sec(show_tpts(end)));
show_dat=se_ftrs(:,ftr_tpts);
full_show_dat=se_ftrs(:,ftr_tpts);
mx_ftr=max(max(full_show_dat));
fprintf('Max ftr value shown: %f z\n',mx_ftr);
min_ftr=min(min(full_show_dat));
fprintf('Min ftr value shown: %f z\n',min_ftr);
imagesc(show_dat);

rel_se_stim_sec=se_time_sec-onset_sec;
xtick=[];
xticklabel=[];
ct=0;
clear onset_x
for tloop=-4:2:14,
    disp(tloop);
    xtick=[xtick findTpt(tloop,rel_se_stim_sec)];
    if tloop==0,
        onset_x=xtick(end);
    end
    ct=ct+1;
    xticklabel{ct}=num2str(tloop);
end
set(gca,'xtick',xtick,'xticklabel',xticklabel,'ytick',[]);
ylim=get(gca,'ylim');
set(gca,'fontsize',fontsize);
hold on;
plot([1, 1]*onset_x,ylim,'r:','linewidth',3);


% Add colorbar
pos=[.91 .11 .02 .815];
limits=[-1, 1];
cmapName='parula';
units='';
nTick=0;
fontSize=12;
unitLocation='top';
hCbar = cbarDGplus(pos,limits,cmapName,nTick,units,unitLocation,fontSize);

if print_em,
    fprintf('Saving figure to edm_example_smoothed_nrg\n');
    set(gcf,'paperpositionmode','auto');
    print(3,'-djpeg','edm_example_smoothed_nrg');
else
    fprintf('Not saving figures!\n');
end


%%
% ax2=subplot(312); 
% plot(se_time_sec,se_szr_class,'r-'); hold on;
% plot(se_time_sec,yhat,'b--')2
% axis tight;


if 0
ax2_pos=[0.1300 0.43  0.7750  0.533];
% ax2=subplot(212);
ax2=axes('position',ax2_pos);
ftr_tpts=find(se_time_sec<=targ_raw_ieeg_sec(show_tpts(end)));
mx_ftr=max(max(se_ftrs(:,ftr_tpts)));
fprintf('Max ftr value shown: %f z\n',mx_ftr);
min_ftr=min(min(se_ftrs(:,ftr_tpts)));
fprintf('Min ftr value shown: %f z\n',min_ftr);
imagesc(se_ftrs(:,ftr_tpts));

%xtick=get(gca,'xtick');
%%%%%% rel_se_stim_sec=se_time_sec-onset_sec;
% xtick=[];
% for tloop=-4:2:14,
%     disp(tloop);
%     xtick=[xtick findTpt(tloop,rel_se_stim_sec)];
% end
set(gca,'xtick',[],'ytick',[]);
% set(gca,'xtick',xtick,'xticklabel',round(rel_se_stim_sec(xtick)),'ytick',[]);
% h=plot(se_time_sec,se_ftrs);
% for floop=1:length(h),
%     clickText(h(floop),ftr_labels{floop},'none');
% end
% axis tight

% Add colorbar
pos=[.91 .43 .02 .533];
limits=[-1, 1];
cmapName='parula';
units='';
nTick=0;
fontSize=12;
unitLocation='top';
hCbar = cbarDGplus(pos,limits,cmapName,nTick,units,unitLocation,fontSize);

end

%% Plot raw iEEG FIGURE 1

% ax1_pos=[0.1300 0.1 0.7750 0.3412];
figure(1); clf();
set(gcf,'position',[131   416   868   276]);
% ax1=subplot(211);
% ax1=axes('position',ax1_pos);
h=plot(targ_raw_ieeg_sec(show_tpts)-onset_sec,targ_raw_ieeg(show_tpts),'k-'); 
set(h,'linewidth',2);
hold on;
% axis tight;
%ylim=get(gca,'ylim');
ylim=[-750, 1300];
set(gca,'fontsize',fontsize);
set(gca,'ylim',ylim,'xlim',[targ_raw_ieeg_sec(1) targ_raw_ieeg_sec(show_tpts(end))]-onset_sec);
%plot([1, 1]*onset_sec,ylim,'r-');
plot([0, 0],ylim,'r:','linewidth',3);
% title(ftr_fname);
ht=ylabel('Voltage');
set(ht,'fontsize',fontsize);
set(gca,'ytick',[],'yaxislocation','right');
h=xlabel('Seconds');
set(gca,'xlim',[show_time_wind(1) show_time_wind(end)])
%show_tpts=rel_se_stim_sec(ftr_tpts);


if print_em,
    fprintf('Saving figure to edm_example_chan\n');
    set(gcf,'paperpositionmode','auto');
    print(1,'-depsc','edm_example_chan');
else
   fprintf('Not saving figures!\n'); 
end


%%
% set(gcf,'paperpositionmode','auto');
% print(1,'-depsc','edm_example');
% print(1,'-djpeg','edm_example');


%% Plot colorbar
% figure(2);
% clf();
% % cbarDG('vert');
% pos=[.1 .1 .1 .7];
% limits=[-1, 1];
% cmapName='parula';
% units='';
% nTick=0;
% fontSize=12;
% unitLocation='top';
% hCbar = cbarDGplus(pos,limits,cmapName,nTick,units,unitLocation,fontSize);
% set(hCbar,'xticklabel',[]);
% set(gcf,'paperpositionmode','auto');
% print(1,'-depsc','pstr_cbar');
% print(1,'-djpeg','pstr_cbar');