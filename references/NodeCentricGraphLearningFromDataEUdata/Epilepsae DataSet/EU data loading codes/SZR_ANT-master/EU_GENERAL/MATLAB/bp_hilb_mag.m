function [hilb_env, hilb_ifreq]=bp_hilb_mag(bp_data,n_step,wind_len_tpts, wind_step_tpts)
%function [hilb_env, hilb_ifreq]=bp_hilb_mag(bp_data,n_step,wind_len_tpts, wind_step_tpts)
%
% Computes hilbert transform magnitude and ?? using a moving window
%
% Inputs:
%  bp_data - band passed EEG data
%  n_step - # of moving windows
%  wind_len_tpts - length of moving window in time pts
%  wind_step_tpts - # of time pts to increment moving window at each step
%
% Ouputs:
%  hilb_env - envelope of hilbert transform (i.e., signal magnitude)
%  hilb_ifreq - ?? some measure that will help detect chirps, work in
%  progress

%_, hilb_ifreq, sgram_sec=ief.bp_hilb_mag
hilb_env=zeros(1,n_step,'single');
hilb_ifreq=NaN;
wind_ids=1:wind_len_tpts;
for a=1:n_step,
    temp_hilb=hilbert(bp_data(wind_ids));
    hilb_env(a)=mean(abs(temp_hilb));
    wind_ids=wind_ids+wind_step_tpts;
end
