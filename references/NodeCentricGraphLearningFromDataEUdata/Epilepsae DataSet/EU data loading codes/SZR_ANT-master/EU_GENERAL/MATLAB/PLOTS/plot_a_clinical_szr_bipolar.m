function plot_a_clinical_szr_bipolar(sub_id,szr_id,start_sec,stop_sec,model_name)
%function plot_a_clinical_szr_bipolar(sub_id,szr_id,start_sec,stop_sec,mode_name)
%
% Required Inputs:
%  sub_id - the subject numeric id (e.g., 970)
%  szr_id - the szr id (1=the first clinical szr)
%  start_sec - the number of seconds before onset to plot
%  stop_sec - the number of seconds after offset to plot
%
% Optional Inputs:
%  model_name - the name of the classifier whose results you would like to
%               plot under the raw iEEG traces. Red * indicate hypothetical 
%               stimulatio ntimes. If not specified this subplot will not
%               be produced.
%
% Examples:
%   plot_a_clinical_szr_bipolar(970,8,30,5);
%   plot_a_clinical_szr_bipolar(1125,1,60,5,'svmAesFinale_1');
%
% This script imports a single clinical szr from a patient and plots them 
% with onset, offset, SOZ channels, and target window on them.
% This script uses bipolar montages
% It starts plotting 15 seconds before clinician onset until 5 sec after the 
% end of the seizure 
% It plots clinician onset and offset in red
% It plots borders of an early onset wind (5 sec before to 10 sec after
% onset) in blue
% Bad channels according to files in ths directory
% /home/dgroppe/GIT/SZR_ANT/EU_METADATA/BAD_CHANS are ignored

if nargin<5,
    plot_yhat=0; 
else
   plot_yhat=1; 
end
   
cli_szr_info=get_szr_fnames(sub_id);
%cli_szr_info=get_szr_fnames(sub_id,ieeg_dir);
n_szrs=length(cli_szr_info);


%%
bipolar_labels=derive_bipolar_pairs(sub_id);
n_chan=size(bipolar_labels,1);


%% Import bad channel names
bad_chan_fname=sprintf('bad_chans_%d.txt',sub_id);
bad_chan_path='/home/dgroppe/GIT/SZR_ANT/EU_METADATA/BAD_CHANS';
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

%% Load csv file of onset and offset time of each clip
on_off_csv_fname=sprintf('/home/dgroppe/GIT/SZR_ANT/EU_METADATA/IEEG_ON_OFF/data_on_off_FR_%d.csv', ...
    sub_id);
on_off=csv2Cell(on_off_csv_fname,',',1);

%%
if szr_id>n_szrs,
   error('szr_id needs to be <= n_szrs which is %d',n_szrs); 
elseif szr_id<1,
    error('szr_id needs to be > 0');
end
for sloop=szr_id:szr_id,
    fprintf('Importing szr #%d\n',sloop);
   
    % Find onset and offset of file
    slash_ids=find(cli_szr_info(sloop).clinical_fname=='/');
    raw_fname=cli_szr_info(sloop).clinical_fname(slash_ids(end)+1:end);
    temp_id=find(raw_fname=='.');
    raw_stem=raw_fname(1:(temp_id-1));
    on_off_id=findStrInCell([raw_stem '.head'],on_off(:,3),1);
    clip_onset_sec=str2num(on_off{on_off_id,4}); % onset of clip in seconds relative to common date (Jan 1, 2000 I think)
    
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
%     targ_window=zeros(pat.a_n_samples,1,'int8');
%     targ_onset_tpt=fszr_onset_tpt-round(Fs*5);
%     if targ_onset_tpt<0,
%         targ_onset_tpt=1;
%     end
%     targ_offset_tpt=fszr_onset_tpt+round(Fs*10);
%     if targ_offset_tpt>pat.a_n_samples,
%         targ_offset_tpt=pat.a_n_samples;
%     end
%     targ_window(targ_onset_tpt:targ_offset_tpt)=1;
    
    
    %% Import data
    preonset_tpts=Fs*start_sec; % preonset baseline
    clip_onset_tpt=fszr_onset_tpt-preonset_tpts; %time pt at which to START data import
    if clip_onset_tpt<1,
        clip_onset_tpt=1;
    end
    
    % Get time pt at which to STOP data import
    postonset_tpts=Fs*stop_sec; % postonset window to show
    clip_offset_tpt=fszr_offset_tpt+postonset_tpts;
    if clip_offset_tpt>pat.a_n_samples,
        clip_offset_tpt=pat.a_n_samples;
    end
    if fszr_offset_tpt>pat.a_n_samples,
        % Szr extends beyond end of file
        offset_missed=1;
        warning('Offset extends beyond end of file!');
    else
       offset_missed=0; 
    end
    clip_szr_class=szr_class(clip_onset_tpt:clip_offset_tpt);
    %clip_targ_wind=targ_window(clip_onset_tpt:clip_offset_tpt);
    
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
    voffset=500; % voltage scaling factor (bigger=more space between chans)
    ieeg_tpt=size(ieeg,2);
    onset_id=min(find(clip_szr_class>0));
    if ~plot_yhat,
      figure(sloop); clf;
    else
       figure(sloop); clf;
       ax1=axes('position',[0.1300 0.1700 0.7750 0.770]);
    end
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
    h=plot(time_sec(onset_id)*[1, 1],ylim,'r--','linewidth',2);
    clickText(h,'Szr Onset');
    if offset_missed==0,
    h=plot(time_sec(offset_id)*[1, 1],ylim,'r--','linewidth',2);
    clickText(h,'Szr Offset');
    end
    
    % Plot target stimulation window onset and offset as blue dashed line
%     targ_ids=find(clip_targ_wind==1);
%     targ_onset_id=min(targ_ids);
%     targ_offset_id=max(targ_ids);
%     h=plot(time_sec(targ_onset_id)*[1, 1],ylim,'b--','linewidth',1);
%     clickText(h,'Target Onset');
%     h=plot(time_sec(targ_offset_id)*[1, 1],ylim,'b--','linewidth',1);
%     clickText(h,'Target Offset');
    % set(gca,'ytick',0:(n_chan-1)*voffset);
    %plot(1:ieeg_tpt,clip_szr_class*1000000,'r-','linewidth',4);
    % set(gca,'xlim',[0 Fs*20]);
    if ~plot_yhat,
        xlabel('Sec');
    else
       set(gca,'xticklabel',[]); 
    end
    ht=title(sprintf('FR_%d: Clin Szr %d in %s',sub_id,sloop,raw_fname));
    set(ht,'interpreter','none')
    axis tight;
  
    % Plot model output and stimulations if requested
    if plot_yhat,
        yhat_dir=sprintf('/home/dgroppe/EU_YHAT/%d_%s/',sub_id,model_name);
        yhat_fname=fullfile(yhat_dir,[raw_stem '_yhat.mat']);
        load(yhat_fname);
        %112500102_0000_yhat.mat')
        
       ax2=axes('position',[0.1300 0.1 0.7750 0.06]);
       yhat_sec=yhat_sec-cli_szr_info(sloop).clinical_onset_sec; % make t=0, szr onset
       %plot(yhat_sec,max(yhat_soz_chans,[],1));
       h=area(yhat_sec,max(yhat_soz_chans,[],1));
       set(h,'facecolor',[1 1 1]*0.5,'edgecolor','k');
       hold on;
       
       % Load hypothetical stimulations
       stim_path=fullfile('/home/dgroppe/GIT/SZR_ANT/MODELS/',model_name);
       load(fullfile(stim_path,sprintf('%d_thresh_0-5_refract_30_stim_results.mat', ...
           sub_id)),'stim_sec');
       % Plot any hypothetical stimulations in this time window
       stim_sec=stim_sec-clip_onset_sec-cli_szr_info(sloop).clinical_onset_sec;
       stim_sec_ids=find( ((stim_sec)>=xlim(1)).* ((stim_sec)<=xlim(2)));
       if isempty(stim_sec_ids)
           fprintf('No stimulations in this time window.\n');
       else
           plot(stim_sec(stim_sec_ids),0.5,'r*');
       end
       
       set(gca,'xlim',xlim,'ylim',[0, 1]);
       plot(xlim,[0.5, 0.5],'r--');
       xlabel('Seconds');
       linkaxes([ax1, ax2],'x');
       
       ylabel('p(stim)');
    end
  
    fprintf('Done with szr %d/%d\n',sloop,n_szrs);
    
end


