classdef normalizedFFT
   
    properties
        
    end
    
    methods
        function [FFT_W, conv_sizes]= my_apply(self, X_raw, load_core)
            Fs = load_core.Fs;
            win_length = ceil(load_core.welchs_win_len * Fs);
            X = createRollingWindow(X_raw, win_length, ceil(load_core.welchs_stride*Fs));
            welches_win_num = size(X,3);
           
            X = permute(X, [4,1,2,3]); % np.swapaxes(np.swapaxes(X, 0, 1), 1, 2);
             
            L = size(X,1);                   % Length of signal
%             T = 1/Fs;                       % Sampling period       
%             t = (0:L-1)*T;                  % Time vector
        
            f_signal = fft(X);
            f_signal = f_signal./repmat(sqrt(sum(abs(f_signal).^2, 4)),[1,1,1,welches_win_num]);
            W = Fs*(0:(L/2))/L;
%             P2 = abs(Y/L);
%             P1 = P2(1:L/2+1);
%             P1(2:end-1) = 2*P1(2:end-1);
            
            conv_sizes = [];
            all_sizess = zeros(size(W));
            for i=1:size(load_core.freq_bands,1)
                lowcut = load_core.freq_bands(i,1);
                highcut = load_core.freq_bands(i,2);
                sizess = zeros(size(W));
                sizess(W<highcut) = 1;
                sizess(W<lowcut) = 0;
                all_sizess = all_sizess + sizess;
                conv_sizes = [conv_sizes sum(sizess)];
            end
            otherdims = repmat({':'},1,ndims(f_signal)-1);

            in_FFT_W = f_signal(squeeze(find(all_sizess==1)),otherdims{:});
            in_FFT_W = permute(in_FFT_W,[2,3,1,4]);
            othersizes = size(in_FFT_W);
            othersizes = othersizes(1:ndims(in_FFT_W)-2);
            FFT_W = in_FFT_W; % reshape(in_FFT_W, [othersizes, size(in_FFT_W,3)*size(in_FFT_W,4)]);
        end
    end
    
end

