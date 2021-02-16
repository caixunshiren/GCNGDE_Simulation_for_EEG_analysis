% f_GetSignalStrFromCells(pv_ChannCell)
% 
function [str_SignalStr] = ...
    f_GetSignalStrFromCells(...
    pv_ChannCell)

    if nargin < 1
        return;
    end
    
    str_SignalStr = '';
    
    for s_Counter = 1:length(pv_ChannCell)
        if s_Counter > 1
            str_SignalStr = sprintf('%s,', str_SignalStr);
        end
        str_SignalStr = sprintf('%s%s', str_SignalStr, pv_ChannCell{s_Counter});
    end
    
return;
