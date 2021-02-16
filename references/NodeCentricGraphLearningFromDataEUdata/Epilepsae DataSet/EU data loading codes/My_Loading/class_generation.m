function [szr_class] = class_generation(ieeg, file, str_onset, str_offset, load_core) %str: sub_, clin_, or nothing

szr_class=zeros(1,size(ieeg,2));
% targ_window=zeros(1,length(ieeg)); % Same as szr class but extended
onset_sec = file.(genvarname(str_onset));
offset_sec = file.(genvarname(str_offset));

% % There are szrs in this file (clinical and/or subclinical)
% for sloop=1:length(onset_sec),
%     onset_id=findTpt(onset_sec(sloop),ieeg_time_sec_pre_decimate);
%     if ~isempty(offset_sec),
%         % Sadly, some szrs have marked onsets but not offsets
%         % When this happens make szr last until end of clip
%         offset_id=findTpt(offset_sec(sloop),ieeg_time_sec_pre_decimate);
%     else
%         offset_id=length(ieeg);
%     end
%     szr_class(onset_id:offset_id)=1;
%     targ_onset_id=onset_id % -Fs*5; %extend 5 seconds in past to try to stimulate before onset -->changed
%     if targ_onset_id<1,
%         targ_onset_id=1;
%     end
%     targ_window(targ_onset_id:offset_id)=1;
% end
ieeg_time_sec_pre_decimate=[0:(size(ieeg,2)-1)]/load_core.Fs;
for sloop=1:length(onset_sec)
    inn_onset_sec = onset_sec(sloop);
    inn_preonset_sec = max(inn_onset_sec - load_core.pre_szr_sec, 0);
    fszr_preonset_tpt = findTpt(inn_preonset_sec,ieeg_time_sec_pre_decimate);
    if ~isempty(offset_sec),
        % Sadly, some szrs have marked onsets but not offsets
        % When this happens make szr last until end of clip
        inn_offset_sec = offset_sec(sloop);
        fszr_offset_tpt = findTpt(inn_offset_sec,ieeg_time_sec_pre_decimate);
    else
        fszr_offset_tpt=size(ieeg,2);
    end
    if(contains(str_onset, 'clinical'))
        fszr_onset_tpt=round(load_core.Fs*inn_onset_sec);
        fszr_offset_tpt = round(load_core.Fs*inn_offset_sec);
    elseif(contains(str_onset, 'sub') || contains(str_onset, 'clin'))
        fszr_onset_tpt = findTpt(inn_onset_sec,ieeg_time_sec_pre_decimate);
%     elseif(contains(str_onset, 'file')) % later --> wasn't in Dr. Groppe's interictal/ictal 
%         fszr_onset_tpt = round(inn_onset_sec/load_core.Fs);
%         fszr_offset_tpt = round(inn_offset_sec/load_core.Fs);
    end
    if(contains(str_onset, 'sub'))
        szr_class(fszr_onset_tpt:fszr_offset_tpt) = -1;
    else
        if(contains(load_core.class_gen_mode, 'detection'))
            szr_class(fszr_onset_tpt:fszr_offset_tpt) = 1;
        elseif(contains(load_core.class_gen_mode, 'prediction'))
            szr_class(fszr_onset_tpt:fszr_offset_tpt) = -1;
            szr_class(fszr_preonset_tpt:fszr_onset_tpt-1) = 1;
        elseif(contains(load_core.class_gen_mode, 'pre+det'))
            szr_class(fszr_preonset_tpt:fszr_offset_tpt) = 1;
        end
    end
end

% if(length(load_core.state_win_lengths)<1)
%     state_counter = 2 if load_Core.detection_flag else 1
%     num_winds = list(np.ceil(np.array(load_core.state_win_lengths)/win_len_sec).astype(np.int))
%     end_ind = win_ind
%     for le in num_winds:
%         start_ind = np.max((0,end_ind-le))
%         y[start_ind:end_ind] = state_counter
%         state_counter +=1
%         end_ind = start_ind
%         if(end_ind<=0):
%             break
% end
end

