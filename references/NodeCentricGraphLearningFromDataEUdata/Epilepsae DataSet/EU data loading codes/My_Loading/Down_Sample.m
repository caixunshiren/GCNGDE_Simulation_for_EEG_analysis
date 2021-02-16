function [ieeg, szr_class] = Down_Sample(ieeg, szr_class, load_core)


down_fact=round(load_core.Fs/load_core.target_Fs);
ieeg = downsample(movmean(ieeg, down_fact, 2)',down_fact)';
szr_class = downsample(movmax(szr_class, down_fact, 2)',down_fact)';

% ieeg=decimate(ieeg,down_fact);
% time_dec=zeros(1,length(ieeg));
% targ_win_dec=zeros(1,length(ieeg));
% szr_class_dec=zeros(1,length(ieeg));
% for tloop=1:length(ieeg),
%     time_dec(tloop)=mean(ieeg_time_sec_pre_decimate([1:down_fact] ...
%         +(tloop-1)*down_fact));
%     targ_win_dec(tloop)=mean(targ_window([1:down_fact] ...
%         +(tloop-1)*down_fact));
%     szr_class_dec(tloop)=mean(szr_class([1:down_fact] ...
%         +(tloop-1)*down_fact));
% end

end

