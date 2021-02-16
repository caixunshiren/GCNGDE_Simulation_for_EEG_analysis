%% This script imports just the clinical szrs from a patient and plots them 
% with onset, offset, SOZ channels, and target window on them.
% This script uses bipolar montages
% It starts plotting 15 seconds before clinician onset until 5 sec after the 
% end of the seizure 
% It plots clinician onset and offset in red
% It plots borders of an early onset wind (5 sec before to 10 sec after
% onset) in blue
% Bad channels according to files in ths directory
% /home/dgroppe/GIT/SZR_ANT/EU_METADATA/BAD_CHANS are ignored
%
% Pics are saved to:
%  SZR_ANT/EU_METADATA/SZR_FIGS/*

%sub_id=1096;
% sub_id=620;
%sub_id=264;
% sub_id=590;
%sub_id=253;
%sub_id=862;
%sub_id=565;
%sub_id=273;
%sub_id=1125;
%sub_id=1077;
% sub_id=818;
%subs=[115, 253, 442, 565, 590, 620]; DONE
%subs=[273, 862, 1077, 970]; % need to redo
%subs=[958, 970];
%subs=[922];
subs=1096;

for sub_id=subs,
    
cli_szr_info=get_szr_fnames(sub_id);
%cli_szr_info=get_szr_fnames(sub_id,ieeg_dir);
n_szrs=length(cli_szr_info);
sub_ct=0;

sub_ct=sub_ct+1;

%%
bipolar_labels=derive_bipolar_pairs(sub_id);
n_chan=size(bipolar_labels,1);

%% Import bad channels
bad_chan_fname=sprintf('bad_chans_%d.txt',sub_id);
if ismac,
    bad_chan_path='/Users/davidgroppe/PycharmProjects/SZR_ANT/EU_METADATA/BAD_CHANS';
else
    bad_chan_path='/home/dgroppe/GIT/SZR_ANT/EU_METADATA/BAD_CHANS';
end
bad_chan_csv=csv2Cell(fullfile(bad_chan_path,bad_chan_fname));

bad_chan_ids=[];
if ~strcmpi(bad_chan_csv{1,1},'None'),
n_bad_chan=size(bad_chan_csv,1);
for a=1:n_bad_chan,
    found=0;
    split_id=find(bad_chan_csv{a}=='-');
    chan1=bad_chan_csv{a}(1:split_id-1);
    chan2=bad_chan_csv{a}(split_id+1:end);
    temp_id=findStrInCell(chan1,bipolar_labels(:,1));
    if strcmpi(chan2,bipolar_labels{temp_id,2}),
       bad_chan_ids=[bad_chan_ids temp_id]; 
       found=1;
    end
    if found==0,
       error('Chan %s is bad according to bad_chans*.txt file, but not found in bipolar_labels', ...
           bad_chan_csv{a});
    end
end
fprintf('Removing %d bad channels.\n',length(bad_chan_ids));
good_chan_ids=setdiff(1:n_chan,bad_chan_ids);
bipolar_labels=bipolar_labels(good_chan_ids,:);
n_chan=size(bipolar_labels,1);
else
   fprintf('No bad channels to remove.\n'); 
end

%%
endless_szrs{sub_ct}=[];
for sloop=1:n_szrs,
%for sloop=1:12, % ??
    %ieeg_fname='/Volumes/ValianteLabEuData/EU/inv/pat_FR_1096/adm_1096102/rec_109600102/109600102_0000.data';
    %     data_fname='109600102_0000.data';
    %     ieeg_fname=fullfile('/Volumes/ValianteLabEuData/EU/inv/pat_FR_1096/adm_1096102/rec_109600102/',data_fname);
    
    fprintf('Importing szr #%d\n',sloop);
    
    % Read header
    pat=bin_file(cli_szr_info(sloop).clinical_fname);
    Fs=pat.a_samp_freq;
    if isempty(Fs),
       error('Could not find file: %s',cli_szr_info(sloop).clinical_fname);
    end
    fprintf('FS=%f\n',Fs);
    fprintf('# of monopolar chans %d\n',pat.a_n_chan);
    fprintf('# of samples=%d\n',(pat.a_n_samples));
    
    %% Identify onset and offset of time windows of interest
    fszr_onset_tpt=round(Fs*cli_szr_info(sloop).clinical_onset_sec);
    fszr_offset_tpt=round(Fs*cli_szr_info(sloop).clinical_offset_sec);
    fprintf('Szr onset tpt %d\n',fszr_onset_tpt);
    fprintf('Szr offset tpt %d\n',fszr_offset_tpt);
    szr_class=zeros(pat.a_n_samples,1,'int8');
    szr_class(fszr_onset_tpt:fszr_offset_tpt)=1;
    
    % Identify early onset wind 5 sec before to 10 sec after onset
    targ_window=zeros(pat.a_n_samples,1,'int8');
    targ_onset_tpt=fszr_onset_tpt-round(Fs*5);
    if targ_onset_tpt<0,
        targ_onset_tpt=1;
    end
    targ_offset_tpt=fszr_onset_tpt+round(Fs*10);
    if targ_offset_tpt>pat.a_n_samples,
        targ_offset_tpt=pat.a_n_samples;
    end
    targ_window(targ_onset_tpt:targ_offset_tpt)=1;
    
    
    %% Import data
    preonset_tpts=Fs*15; % 15 second preonset baseline
    clip_onset_tpt=fszr_onset_tpt-preonset_tpts; %time pt at which to START data import
    if clip_onset_tpt<1,
        clip_onset_tpt=1;
    end
    
    % Get time pt at which to STOP data import
    postonset_tpts=Fs*5; % 5 second postonset baseline
    clip_offset_tpt=fszr_offset_tpt+postonset_tpts;
    if clip_offset_tpt>pat.a_n_samples,
        clip_offset_tpt=pat.a_n_samples;
    end
    if fszr_offset_tpt>pat.a_n_samples,
        % Szr extends beyond end of file
        offset_missed=1;
        warning('Offset extends beyond end of file!');
        endless_szrs{sub_ct}=[endless_szrs{sub_ct} sloop];
    else
       offset_missed=0; 
    end
%     if fszr_offset_tpt>pat.a_n_samples,
%         clip_offset_tpt=pat.a_n_samples;
%         offset_missed=1;
%     else
%         clip_offset_tpt=fszr_offset_tpt;
%         offset_missed=0;
%     end
    clip_szr_class=szr_class(clip_onset_tpt:clip_offset_tpt);
    clip_targ_wind=targ_window(clip_onset_tpt:clip_offset_tpt);
    
    ieeg_labels=cell(n_chan,1);
    for chan_loop=1:n_chan,
        pat.a_channs_cell={bipolar_labels{chan_loop,1}}; % Channels to import
        %ieeg(1:n_chan,:)=pat.get_bin_signals([],[]);
        ieeg_temp1=pat.get_bin_signals(clip_onset_tpt,clip_offset_tpt);
        
        pat.a_channs_cell={bipolar_labels{chan_loop,2}}; % Channels to import
        ieeg_temp2=pat.get_bin_signals(clip_onset_tpt,clip_offset_tpt);
        
        if chan_loop==1,
            n_tpt=size(ieeg_temp1,2);
            ieeg=zeros(n_chan,n_tpt);
        end
        ieeg(chan_loop,:)=ieeg_temp1-ieeg_temp2;
        ieeg_labels{chan_loop,1}=[bipolar_labels{chan_loop,1} '-' bipolar_labels{chan_loop,2}];
    end
    
    %% Strat plot
    voffset=500;
    ieeg_tpt=size(ieeg,2);
    onset_id=min(find(clip_szr_class>0));
    figure(sloop); clf;
    onset_chans=cli_szr_info(sloop).clinical_soz_chans;
    time_sec=([1:ieeg_tpt]-onset_id)/Fs;
    for cloop=1:n_chan,
        indiv_chans=strsplit(ieeg_labels{cloop},'-');
        onset_chan=0;
        for biloop=1:2,
            if ~isempty(findStrInCell(indiv_chans{biloop},onset_chans)),
                onset_chan=1;
            end
        end
        if onset_chan,
            h=plot(time_sec,ieeg(cloop,:)+(cloop-1)*voffset,'m-');
        else
            h=plot(time_sec,ieeg(cloop,:)+(cloop-1)*voffset,'k-');
        end
        clickText(h,ieeg_labels{cloop});
        hold on;
    end
    ylim=[-voffset, voffset*n_chan];
    xlim=get(gca,'xlim');
    %ylim=get(gca,'ylim');
    set(gca,'ylim',ylim,'ytick',[]);
    % plot([1, 1]*Fs*10,ylim,'r-');
    szr_ids=find(clip_szr_class==1);
    onset_id=min(szr_ids);
    offset_id=max(szr_ids);
    % plot clinician ONSET & OFFSET as red dashed line
    h=plot(time_sec(onset_id)*[1, 1],ylim,'r--','linewidth',1);
    clickText(h,'Szr Onset');
    if offset_missed==0,
    h=plot(time_sec(offset_id)*[1, 1],ylim,'r--','linewidth',1);
    clickText(h,'Szr Offset');
    end
    
    % Plot target stimulation window onset and offset as blue dashed line
    targ_ids=find(clip_targ_wind==1);
    targ_onset_id=min(targ_ids);
    targ_offset_id=max(targ_ids);
    h=plot(time_sec(targ_onset_id)*[1, 1],ylim,'b--','linewidth',1);
    clickText(h,'Target Onset');
    h=plot(time_sec(targ_offset_id)*[1, 1],ylim,'b--','linewidth',1);
    clickText(h,'Target Offset');
    % set(gca,'ytick',0:(n_chan-1)*voffset);
    %plot(1:ieeg_tpt,clip_szr_class*1000000,'r-','linewidth',4);
    % set(gca,'xlim',[0 Fs*20]);
    xlabel('Sec');
    slash_ids=find(cli_szr_info(sloop).clinical_fname=='/');
    raw_fname=cli_szr_info(sloop).clinical_fname(slash_ids(end)+1:end);
    ht=title(sprintf('FR_%d: Clin Szr %d in %s',sub_id,sloop,raw_fname));
    set(ht,'interpreter','none')
    axis tight;
    fprintf('Done with szr %d/%d\n',sloop,n_szrs);

    %% Save figure
    if ismac,
        fig_path=fullfile('/Users/davidgroppe/PycharmProjects/SZR_ANT/EU_METADATA/SZR_FIGS',num2str(sub_id));
    else
        fig_path=fullfile('/home/dgroppe/GIT/SZR_ANT/EU_METADATA/SZR_FIGS',num2str(sub_id));
    end
    if ~exist(fig_path,'dir')
        mkdir(fig_path);
    end
    fig_fname=fullfile(fig_path,sprintf('strat_szr%d.fig',sloop));
    savefig(sloop,fig_fname,'compact');
    
    %% Butterfly plot
%     figure(2); clf;
%     plot(1:ieeg_tpt,ieeg'); hold on;
%     xlim=get(gca,'xlim');
%     ylim=get(gca,'ylim');
%     % plot([1, 1]*Fs*10,ylim,'r-');
%     plot(1:ieeg_tpt,single(clip_szr_class)*ylim(2),'r-','linewidth',4);
%     %plot(1:ieeg_tpt,clip_szr_class*1000000,'r-','linewidth',4);
%     % set(gca,'xlim',[0 Fs*20]);
%     axis tight;
    
end
end

disp('Done!!');

