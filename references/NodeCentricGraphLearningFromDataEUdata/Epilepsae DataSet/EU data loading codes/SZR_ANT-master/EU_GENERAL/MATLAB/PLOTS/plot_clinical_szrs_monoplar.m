%% This script imports just the clinical szrs from a patient and plots them with onset, offset, SOZ channels, and target
% window on them.
% This script use MONOpolar (i.e., the recording reference). I wrote it
% because the bipolar SOZ seemed to disagree with what it was supposed to
% be according to the database.

%sub_id=1096;
% sub_id=620;
%sub_id=264;
sub_id=590;
cli_szr_info=get_szr_fnames(sub_id);
%cli_szr_info=get_szr_fnames(sub_id,ieeg_dir);
n_szrs=length(cli_szr_info);

%%
monopolar_labels=derive_monopolar_pairs(sub_id);
n_chan=size(monopolar_labels,1);

%%
for sloop=1:n_szrs,
%for sloop=1:1,
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
    
    % Get data from 30 seconds before to end of szr
    
    
    %%
    fszr_onset_tpt=round(Fs*cli_szr_info(sloop).clinical_onset_sec);
    fszr_offset_tpt=round(Fs*cli_szr_info(sloop).clinical_offset_sec);
    fprintf('Szr onset tpt %d\n',fszr_onset_tpt);
    fprintf('Szr offset tpt %d\n',fszr_offset_tpt);
    szr_class=zeros(pat.a_n_samples,1,'int8');
    szr_class(fszr_onset_tpt:fszr_offset_tpt)=1;
    
    % Identify target window for classifier 5 sec before to 10 sec aftet
    % onset
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
    
    
    %% 
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
        pat.a_channs_cell={monopolar_labels{chan_loop,1}}; % Channels to import
        %ieeg(1:n_chan,:)=pat.get_bin_signals([],[]);
        ieeg_temp1=pat.get_bin_signals(clip_onset_tpt,clip_offset_tpt);
        
        if chan_loop==1,
            n_tpt=size(ieeg_temp1,2);
            ieeg=zeros(n_chan,n_tpt);
        end
        ieeg(chan_loop,:)=ieeg_temp1;
        ieeg_labels{chan_loop,1}=monopolar_labels{chan_loop,1};
    end
    
    %% Strat plot
    voffset=1000;
    ieeg_tpt=size(ieeg,2);
    figure(sloop); clf;
    onset_chans=cli_szr_info(sloop).clinical_soz_chans;
    time_sec=[1:ieeg_tpt]/Fs;
    for cloop=1:n_chan,
        if ~isempty(findStrInCell(ieeg_labels{cloop},onset_chans)),
            onset_chan=1;
        else
            onset_chan=0;
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
    h=plot(time_sec(onset_id)*[1, 1],ylim,'r--','linewidth',1);
    clickText(h,'Szr Onset');
    h=plot(time_sec(offset_id)*[1, 1],ylim,'r--','linewidth',1);
    clickText(h,'Szr Offset');
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
    ht=title(sprintf('FR_%d: Clin Szr %d',sub_id,sloop));
    set(ht,'interpreter','none')
    axis tight;

    fprintf('Done with szr %d/%d\n',sloop,n_szrs);

    %% Save figure
    fig_path=fullfile('SZR_FIGS',num2str(sub_id));
    if ~exist(fig_path,'dir')
        mkdir(fig_path);
    end
    fig_fname=fullfile(fig_path,sprintf('strat_szr%d_mono.fig',sloop));
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


disp('Done!!');

