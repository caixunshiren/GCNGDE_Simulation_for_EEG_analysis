function data=butterfilt4_causal(data,srate,flt,n_pad)
% butterfilt4_causal() - Filters EEG data with low pass, high pass, band-pass,
%                or band-stop 4th order Butterworth filter.  
%
% Usage:
%  >> data=butterfilt4_causal(data,srate,flt,n_pad);
%
% Required Inputs:
%   data      - 2 dimensional (channel x time point) matrix of EEG data
%   srate     - Sampling rate (in Hz)
%   filt      - [low_boundary high_boundary] a two element vector indicating the frequency
%               cut-offs for a 4th order Butterworth filter that will be applied to each
%               trial of data.  If low_boundary=0, the filter is a low pass filter.  If
%               high_boundary=(sampling rate)/2, then the filter is a high pass filter.  If both
%               boundaries are between 0 and (sampling rate)/2, then the filter is a band-pass filter.
%               If both boundaries are between 0 and -(sampling rate)/2, then the filter is a band-
%               stop filter (with boundaries equal to the absolute values of the low_boundary and
%               high_boundary).  Note, using this option requires the signal processing toolbox
%               function butter.m.  You should probably use the 'baseline' option as well since the
%               mean prestimulus baseline may no longer be 0 after the filter is applied
%
% Optional Inputs:
%   n_pad     - [integer] number of time points to pad each epoch of
%               data before filtering.  For example, if n_pad is 512 and
%               you have 256 time points of data per epoch, each epoch of
%               data will be grown to 1280 time points (i.e., 512*2+256) by
%               adding 512 padding points before and after the 256 time points
%               of data.  The values of the padding points form a line from the
%               value of the first and last data point (i.e., if you wrapped 
%               the padded waveform around a circle the pre-padding and
%               post-padding points would form a line).  This method of
%               padding is the way the Kutaslab function filter.c works.
%               If n_pad is not specified, butterfilt.m will follow the Kutaslab
%               convention of n_pad=2*(the number of time points per
%               epoch). If you wish to avoid zero padding, set n_pad to 0.
%               THIS OPTION IS NOT FULLY SUPPORTED YET
%              
%
% Example:
% 1) A 15 Hz low pass filter with no zero padding
% >> data=butterfiltMK(data,250,[0 15],0);
%
% 2) A .2 Hz high pass filter with default, Kutaslab-like, zero padding.
% Note, the sampling rate in this example is 250 Hz.
% >> data=butterfiltMK(data,250,[0.2 125],2);
%
% 3) A .2 to 15 Hz band pass filter with no zero padding
% >> data=butterfiltMK(data,250,[0.2 15],[-100 0],0);
%
% Author:
% David Groppe
% Kutaslab, 12/2009
%

%%%%%%%%%%%%%%%% REVISION LOG %%%%%%%%%%%%%%%%%
%
% 1/10/2010 - Function now checks for ICA mixing matrix before attempting to
%recompute IC features
%
% 11/5/2010 - n_pad option added
%
% 5/9/2013  - Function adapted from butterfilt.m. The only real change is
% that it no longer requires an EEGLAB EEG variable and I haven't double
% checked to make sure the pad option works.


%%%%%%%%%%%%%%%% FUTURE ADDITIONS %%%%%%%%%%%%%%%%%
%-add verblevel?

if nargin<2,
    error('butterfilt4_causal.m requires at least two inputs.');
end

[n_chans n_pnts]=size(data);
if nargin<3 || isempty(n_pad),
    n_pad=2*n_pnts;
end

if length(flt)~=2,
    error('''filt'' parameter argument should be a two element vector.');
elseif max(flt)>(srate/2),
    error('''filt'' parameters need to be less than or equal to (sampling rate)/2 (i.e., %f).',srate/2);
elseif (flt(2)==(srate/2)) && (flt(1)==0),
    error('If second element of ''filt'' parameter is srate/2, then the first element must be greater than 0.');
elseif abs(flt(2))<=abs(flt(1)),
    error('Second element of ''filt'' parameters must be greater than first in absolute value.');
elseif (flt(1)<0) || (flt(2)<0),
    if (flt(1)>=0) || (flt(2)>=0),
        error('BOTH parameters of ''filt'' need to be greater than or equal to zero OR need to be negative.');
    end
    if min(flt)<=(-srate/2),
        error('''filt'' parameters need to be greater than sampling rate/2 (i.e., -%f) when creating a stop band.',srate/2);
    end
end

fprintf('\nFiltering data with 4th order Butterworth filter: ');
if (flt(1)==0),
    %lowpass filter the data
    [B, A]=butter(4,flt(2)*2/srate,'low');
    fprintf('lowpass at %.2f Hz\n',flt(2));
elseif (flt(2)==(srate/2)),
    %highpass filter the data
    [B, A]=butter(4,flt(1)*2/srate,'high');
    fprintf('highpass at %.2f Hz\n',flt(1));
elseif (flt(1)<0)
    %bandstop filter the data
    flt=-flt;
    [B, A]=butter(4,flt*2/srate,'stop');
    fprintf('stopband from %.2f to %.2f Hz\n',flt(1),flt(2));
else
    %bandpass filter the data
    [B, A]=butter(4,flt*2/srate);
    fprintf('bandpass from %.2f to %.2f Hz\n',flt(1),flt(2));
end
% fprintf('A: ')
% disp(A);
% fprintf('B: ');
% disp(B);

%preallocate memory
total_pad=n_pad*2;
total_tpts=total_pad+n_pnts;
padded=zeros(1,total_tpts);


fprintf('Adding %d time points of zeros before and after each epoch of data.\n',n_pad);
fprintf('Padded values are linear interpolations between the first and last value of each epoch of data.\n');
if n_pad,
    for chan=1:n_chans,
        error('I am not sure if padding option this works yet. Need to double check.');
        fprintf('Now filtering channel: %d of %d\n',chan,n_chans);
        padded(n_pad+1:n_pad+n_pnts)=squeeze(data(chan,:)); %put real data in the middle
        inc=(data(chan,n_pnts)-data(chan,1))/(total_pad+1);
        %add padding before the real data
        for a=n_pad:-1:1,
            padded(a)=padded(a+1)+inc;
        end
        %add padding after the real data
        for a=n_pad+n_pnts+1:total_tpts,
            padded(a)=padded(a-1)-inc;
        end
        %Note, those for loops are actually faster than using the
        %MATLAB linspace function.  Go figure.
        filtered=filtfilt(B,A,padded);
        data(chan,:)=filtered(n_pad+1:n_pad+n_pnts);
    end
else
    %no padding
    for chan=1:n_chans,
        %data(chan,:)=filtfilt(B,A,squeeze(data(chan,:)));
        data(chan,:)=filter(B,A,squeeze(data(chan,:)));
    end
end




%% %%%%%%%%%%%%%%%%%%%%% function find_crspnd_pt() %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function y_pt=find_crspnd_pt(targ,vals,outtrials)
%function id=find_crspnd_pt(targ,vals,outtrials)
%
% Inputs:
%   targ      - desired value of sorting variable
%   vals      - a vector of observed sorting variables (possibly smoothed)
%   outtrials - a vector of y-axis values corresponding to trials in the
%               ERPimage (this will just be 1:n_trials if there's no
%               smoothing)
%
% Output:
%   y_pt  - y-axis value (in units of trials) corresponding to "targ".
%          If "targ" matches more than one y-axis pt, the median point is
%          returned.  If "targ" falls between two points, y_pt is linearly
%          interpolated.
%
% Note: targ and vals should be in the same units (e.g., milliseconds)

%find closest point above
abv=find(vals>=targ);
if isempty(abv),
    %point lies outside of vals range, can't interpolate
    y_pt=[];
    return
end
abv=abv(1);

%find closest point below
blw=find(vals<=targ);
if isempty(blw),
    %point lies outside of vals range, can't interpolate
    y_pt=[];
    return
end
blw=blw(end);

if (vals(abv)==vals(blw)),
    %exact match
    ids=find(vals==targ);
    y_pt=median(outtrials(ids));
else
    %interpolate point
    
    %lst squares inear regression
    B=regress([outtrials(abv) outtrials(blw)]',[ones(2,1) [vals(abv) vals(blw)]']);
    
    %predict outtrial point from target value
    y_pt=[1 targ]*B;
    
end



