function v_Data = ...
    f_LoadI32File( ...
    pstr_FileName, ...
    ps_FirstIndex, ...
    ps_LastIndex)
% 
% Function: f_LoadI32File.m
% 
% Description: 
% This function loads an integer signal from a file where each data value
% is stored in 4 bytes.
% 
% Inputs:
% pstr_FileName: name of the file
% ps_FirstIndex (optional): number of the first sample to load from.
% Default: 1
% ps_LastIndex (optional): number of the last sample to load to. Default:
% end of the file
% 
% Outputs:
% v_Data: loaded data
% 
% MATLAB Version: R2008a
%
%  
%EU FP7 Grant 211713 (EPILEPSIAE)
%
% César Teixeira
% CISUC-FCTUC
% June 2010
%
% Adapted from f_LoadI32File developed by:
% Team: LENA
% Author: Mario Valderrama
%
    if nargin < 1
        error('[f_RWaveDet] - ERROR: bad number of inputs!');
    end

    s_FirstIndex = 1;
    s_LastIndex = -1;
    if nargin >= 2 && ~isempty(ps_FirstIndex)
        s_FirstIndex = ps_FirstIndex;
    end    
    if nargin >= 3 && ~isempty(ps_LastIndex)
        s_LastIndex = ps_LastIndex;
    end    
    
    clear v_Data;
    s_Size = (s_LastIndex - s_FirstIndex) + 1;
    if s_FirstIndex < 1 || (s_LastIndex > 0 && s_Size < 1)
        return;
    end
    s_File = fopen(pstr_FileName, 'r');
    if s_File == -1
        return;
    end
    if s_FirstIndex > 1
        fseek(s_File, 4 * (s_FirstIndex - 1), 'bof');
    end
    if s_LastIndex > 0
        v_Data = fread(s_File, s_Size, 'int32');
    else
        v_Data = fread(s_File, 'int32');
    end
    fclose(s_File);
end
