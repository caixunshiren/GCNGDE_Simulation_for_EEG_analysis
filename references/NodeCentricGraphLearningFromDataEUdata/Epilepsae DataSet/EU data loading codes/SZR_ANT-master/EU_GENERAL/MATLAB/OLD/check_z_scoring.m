% This script was to check that spectral energy (SE) features were being
% properly normalized. They were, but the SE features I plotted for my CLAE
% poster were not (I thought they were).
%
% So classifier accuracy is correct, but poster needs to be changed.

%%
load('/Users/davidgroppe/PycharmProjects/SZR_ANT/EU_GENERAL/EU_GENERAL_FTRS/SE/1096/1096_HL2_HL3_non.mat');
%load('/Users/davidgroppe/PycharmProjects/SZR_ANT/EU_GENERAL/EU_GENERAL_FTRS/SE/1096/1096_HL9_HL10_non.mat');
% nonszr_se_ftrs=log10(nonszr_se_ftrs+1);

% Visualize central tendency & dispersion with boxplots and the like
figure(10); clf;
subplot(3,1,1);
boxplot(nonszr_se_ftrs(1:6,:)');
set(gca,'xticklabels',ftr_labels(1:6),'xtick',1:6);

subplot(3,1,2);
mns=mean(nonszr_se_ftrs(1:6,:),2);
sds=std(nonszr_se_ftrs(1:6,:),1,2);
h=notBoxPlot(nonszr_se_ftrs(1:6,:)'); hold on;
plot(1:6,mns,'ro');
for a=1:6,
   plot(a,mns(a)+[-1 1]*sds(a),'go-','linewidth',2); 
end

z=nonszr_se_ftrs(1:6,:);
for a=1:6,
    z(a,:)=(z(a,:)-mns(a))/sds(a);
end
subplot(3,1,3);
h=notBoxPlot(z(1:6,:)'); hold on;


%% Load a seizure and normalize it

% Load raw ftrs
ftr_root='/Users/davidgroppe/PycharmProjects/SZR_ANT/EU_GENERAL/EU_GENERAL_FTRS/SE/';
sub=1096;
chan='HL2-HL3';
%chan='HL3-HL4';
dash_id=find(chan=='-');
chan1=chan(1:dash_id-1);
chan2=chan(dash_id+1:end);
ftr_fname=sprintf('%d_%s_%s_szr6.mat',sub,chan1,chan2);
fprintf('Loading %s\n',ftr_fname);
%ftr_fname='1096_HL3_HL4_szr6.mat';
load(fullfile(ftr_root,num2str(sub),ftr_fname));

%normalize ftrs
se_ftrs_z=se_ftrs(1:6,:);
for a=1:6,
    se_ftrs_z(a,:)=(se_ftrs_z(a,:)-mns(a))/sds(a);
end

figure(20); clf;
ax1=subplot(211); 
h=plot(se_time_sec,se_ftrs_z(1:6,:),'linewidth',2); hold on;
for floop=1:6,
    clickText(h(floop),ftr_labels{floop});
end
ylabel('Z');
axis tight;

% Load pre-normalized ftrs
load('/Users/davidgroppe/PycharmProjects/SZR_ANT/MODELS/genLogRegSe_1_yhat/1096_HL2_HL3_phat_szr6.mat');
ax1=subplot(212); 
h=plot(se_time_sec,ftrs_z(1:6,:),'linewidth',2); hold on;
for floop=1:6,
    clickText(h(floop),ftr_labels{floop});
end
axis tight;
ylabel('Raw Mag');