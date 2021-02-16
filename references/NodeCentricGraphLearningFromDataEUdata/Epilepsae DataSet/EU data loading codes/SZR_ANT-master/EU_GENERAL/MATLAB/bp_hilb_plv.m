function [hilb_pcos, hilb_psin]=bp_hilb_plv(bp_data1, bp_data2, n_step, wind_len_tpts, wind_step_tpts)
%function hilb_plv=bp_hilb_plv(bp_data1, bp_data2, n_step, wind_len_tpts, wind_step_tpts)
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

%_, hilb_ifreq, sgram_sec=ief.bp_hilb_mag
hilb_psin=zeros(1,n_step,'single');
hilb_pcos=zeros(1,n_step,'single');
wind_ids=1:wind_len_tpts;
for a=1:n_step,
    temp_hilb1=hilbert(bp_data1(wind_ids));
    temp_hilb2=hilbert(bp_data2(wind_ids));
    
    % Extract phase angle from complex output of hilbert transform
    ch1_angle = angle(temp_hilb1); % should use cordicangle.m to make this realistic
    ch2_angle = angle(temp_hilb2);
    
    % Subtract one time series of angles from the other
    angle_dif=ch1_angle-ch2_angle;
     
    cos_angle_dif = cos(angle_dif);
    sin_angle_dif = sin(angle_dif);
    hilb_pcos(a)=mean(cos_angle_dif);
    hilb_psin(a)=mean(sin_angle_dif);
    
    wind_ids=wind_ids+wind_step_tpts;
end
