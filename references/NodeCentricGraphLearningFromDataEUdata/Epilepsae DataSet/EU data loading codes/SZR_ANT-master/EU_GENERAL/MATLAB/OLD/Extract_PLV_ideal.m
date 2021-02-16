function [ PLV_data ] = Extract_PLV_ideal (ch1_in, ch2_in, alphas)

% input_data = BPF_signal;
% PLV Configuration
n_bits = 16;
b_width = 16;

%alphas = 9:14;

n_samples = size(ch1_in,1);

%% Hilbert and angle
% fprintf('Length of signal: %d\n', length(ch1_in));
% fprintf('Hilbert Time : ');
% tic;
input_data_1_cplx = hilbert(ch1_in);
input_data_2_cplx = hilbert(ch2_in);
% toc;

% fprintf('\nArcTan Time: ');
% tic;
ch1_angle = angle(input_data_1_cplx);
ch2_angle = angle(input_data_2_cplx);
% toc;

%     fig = figure(6);
%     clf(fig);
%     hold on;
%     plot(t,A_HF);
%     plot(t,P_LF);

% VERILOG

% 		SINCOS : begin
% 			if ( $signed(dphi_comb) > $signed(`PI_17Q13) )
% 			begin
% 				dphi_next = $signed(dphi_comb) - $signed(`TWOPI_17Q13);
% 			end
% 			else if ( $signed(dphi_comb) < $signed(`PI_NEG_17Q13) ) begin
% 				dphi_next = $signed(dphi_comb) + $signed(`TWOPI_17Q13);
% 			end
% 			else begin
% 				dphi_next = $signed(dphi_comb);
%         end



%TODO
%unwrap(ch1_1_angle)
dphi = ch1_angle-ch2_angle;

% fprintf('Unwrap: ');
% tic;
% unwrap(dphi); % DG: ?? This line currently has no effect
% toc;

%     for k=1:length(dphi)
%
%         if (dphi(k) > pi)
%             dphi(k) = dphi(k) - 2*pi;
%         elseif (dphi(k) < -pi)
%             dphi(k) = dphi(k) + 2*pi;
%         end
%     end
%dphi = unwrap(dphi);



%% Sin Cos dPhi
%fprintf('SimCos dPhi Time: ');
%tic;
plv_4cosdphi = cos(dphi);
plv_4sindphi = sin(dphi);
%toc;

%% MA Filter Approximation
%fprintf('MA Filter Time: ');
% Moving average approximation filter for sine and cosine, where alpha = 1/N
% See http://dsp.stackexchange.com/a/384


% loop through each possible value for alpha
n_alpha = length(alphas);
PLV_data = zeros(n_samples,n_alpha);

ma_n =alphas;

% loop over possible alpha values
for ma_i=1:length(ma_n)
    
    cur_n=ma_n(ma_i);
    alpha = 1/2^cur_n; % For moving average approximation
    
    %fprintf('Processing Shift Value of: %i ', cur_n);
    
    ma1 = [0; 0];
    ma2 = [0; 0];
    
    % Construct PLV time series
%     tic;
    for i=2:length(plv_4cosdphi)
        ma1(i) = ma1(i-1) - alpha*(ma1(i-1) - plv_4cosdphi(i));
        ma2(i) = ma2(i-1) - alpha*(ma2(i-1) - plv_4sindphi(i));
    end
%     toc;
    
    
    %% Final Magnitude calculation
    % fprintf('Final Magnitude calculation: ');
    
    % find magnitude of average sine and cosine vector
    %tic;
    PLV_in = complex(double(ma1),double(ma2));
    PLV = abs(PLV_in);
    %toc;
    
    PLV_data(:,ma_i) = PLV(:)';
    
end


%% Plotting

%     figure(7);
%     pres = plot(PLV,'k');
%     title('PLV');
%

%end