classdef myCorr
    
    properties
        
    end
    
    methods
        function [agg_corrs, conv_sizes]= my_apply(self, X_raw, load_core)
            Fs = load_core.Fs;
            X = permute(X_raw, [3,2,1]);
            celll = squeeze(num2cell(X, [1,2]));
            agg_corrs = cellfun(@corr, celll ,'un',0);
            conv_sizes = NaN;
        end
    end
    
end

