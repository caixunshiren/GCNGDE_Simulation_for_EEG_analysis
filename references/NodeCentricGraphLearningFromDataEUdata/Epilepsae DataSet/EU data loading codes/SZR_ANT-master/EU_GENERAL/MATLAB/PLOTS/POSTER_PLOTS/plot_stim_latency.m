% Script for plotting peri-onset stimualtion latencies in CLAE poster

% TODO color SD bars so that they map to text

% Load stim latencies from file
% load('/Users/davidgroppe/PycharmProjects/SZR_ANT/EU_GENERAL/FOR_POSTER/stim_lat.mat')
% 
% all_lat=[ptnt1096, ptnt1125];
% grp=ones(1,length(all_lat));
% grp(1:length(ptnt1096))=0;
% 
% % Boxplots
% figure(1);
% clf();
% %boxplot(ptnt1096);
% h=boxplot(all_lat,grp,'labels',{'Test1','Test2'},'positions',[1, 1.25], ...
%     'boxstyle','filled');
% set(gca,'fontsize',32,'linewidth',2,'box','off');

%%
% mn_lat=[mean(ptnt1096), mean(ptnt1125)];
% clear rgb
% rgb{1}=[88, 29, 175]/255;
% rgb{2}=[88, 29, 175]/255;
% rgb{3}=[88, 29, 175]/255;
% 
% figure(1);
% clf();
% hold on;
% %h=bar(mn_lat);
% for a=1:2,
%     h=bar(a,mn_lat(a));
%     set(h,'facecolor',rgb{a});
% end
% set(gca,'xtick',[],'ylim',[0.5, 1],'ytick',[0.5:.1:1], ...
%     'fontsize',32,'xlim',[0.5, 2.5]);
% print(gcf,'-depsc','mn_lat');

%%
stim_lat_cell{1}=ptnt1096;
stim_lat_cell{2}=ptnt1125;
mn_lat=[mean(ptnt1096), mean(ptnt1125)];
sd_lat=[std(ptnt1096), std(ptnt1125)];
xpos=[1,2];
xlim=[0.7, 2.3];

prpl=[88, 29, 175]/255;
pnk=[175, 29, 88]/255;
gry=[1, 1, 1]*.3;

figure(1);
clf();
h=plot(xlim,[0, 0],'k--','linewidth',2);
set(h,'color',gry);
hold on;
%h=bar(mn_lat);
for a=1:2,
    plot([1, 1]*a,mn_lat(a)+[-1, 1]*sd_lat(a),'-','linewidth',4, ...
        'color',prpl);
    plot(a+[-1, 1]*0.2,[1, 1]*mean(stim_lat_cell{a}),'r-', ...
        'linewidth',4);
    plot(a+randn(length(stim_lat_cell{a}),1)/100,stim_lat_cell{a},'.b','linewidth',12,'markersize',32);
%     h=bar(a,mn_lat(a));
%     set(h,'facecolor',rgb{a});
end
ht=ylabel('Seconds');
set(gca,'xtick',xpos,'ylim',[-15, 20],'ytick',[-10:10:20], ...
    'fontsize',32,'xlim',xlim,'linewidth',2,'box','on', ...
    'xticklabel',{'Patient 1','Patient 2'});
ht=text(1.5,-1.2,'Seizure Onset','fontsize',32);
set(ht,'HorizontalAlignment','center','color',gry);
% ht=title('Peri-Onset Stimulation Latency');
% set(ht,'fontsize',32);

ht=text(1.05,mn_lat(1)+2,sprintf('%.1f sec',mn_lat(1)),'fontsize',32, ...
    'color','r');
ht=text(1.95,mn_lat(2)+2,sprintf('%.1f sec',mn_lat(2)),'fontsize',32, ...
    'color','r','HorizontalAlignment','right');

print(gcf,'-depsc','mn_lat');