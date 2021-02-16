classdef bin_file < handle
    %======================================================================
    % Matlab class that defines a "bin_file" object.
    %    
    % This class is intended to load and read data stored in raw binary
    % files according to the following specifications:
    % 
    % 1. Each file consists actually in a couple of files: the first one
    % contains the raw data stored in binary values (binary file); the
    % second one corresponds to a text file containing some parameters
    % about how the data was acquired and stored (header file).
    %         
    % 2. The parameters present in the header file (as it was decided in
    % the meeting in Coimbra on July 2009) are the following:
    % 
    % start_ts=<yyyy-mm-dd hh:mm:ss.microseconds>
    % num_samples=<int>
    % sample_freq=<int>
    % conversion_factor=<float>
    % num_channels:=<int>
    % (optional)elec_names=<string>, ...  (e.g. FP1,FP2)
    % (opt.)pat_id=<int>
    %
    % 3. The binary file is organized as follows:
    %
    % - Each sample (data value) in the file is stored in 2 bytes, signed.
    % (“short” type in C, “int16” in MATLAB)
    % - The firsts 2 bytes in the file correspond to the value of the first
    % sample for the first channel, the next 2 to the value of the first
    % sample for the second channel and so on until the first sample for
    % the last channel (according to parameter “num_channels” above). After
    % the first sample for all channels, the next series will be the second
    % sample for each progressive channel and the sequence continues until
    % the end of the file.
    %
    % EU FP7 Grant 211713 (EPILEPSIAE)
    %
    % Mario Valderrama
    % PARIS - CRNRS
    % September 2009
    %======================================================================
    
    properties
        
        a_n_chan; % number of channels
        a_samp_freq; % sampling rate
        a_n_bytes; % number of bytes per sample in the file
        a_n_samples; % total number of samples per channel
        a_n_data_secs; %  total number of seconds per channel
        a_duration_ts; % dd HH:MM:SS.fff
        a_start_ts; % yyyy-mm-dd HH:MM:SS.fff
        a_stop_ts; % yyyy-mm-dd HH:MM:SS.fff
        a_file_elec_cell; % cell array containing the names of all electrodes presented in the file
        a_conv_factor; % conversion factor
        a_note_struct; % structure array containing the notes and the related sample number

        a_step; % step between data windows (in seconds)
        a_wsize; % window size in seconds
        a_channs_cell; % cell array containing the names of electrodes to be processed. The name can contain channels in MONOPOLAR, BIPOLAR or AVERAGE montage
        a_channs_ave_cell; % cell array containing the names of electrodes to be included in the average. Default value: empty
        a_first_data_bytes_offset=0; % offset in bytes of the first sample to be read
        a_first_data_samp_offset; % offset in samples of the first sample to be read
        a_curr_bytes_offset; % offset in bytes from the beginning of the file to the beginning of the current window
        a_last_bytes_offset; % offset in bytes from the beginning of the file to the end of the current window
        a_curr_samp_offset; % offset in samples from the beginning of the file to the beginning of the current window
        a_last_samp_offset; % offset in samples from the beginning of the file to the end of the current window
        a_is_first_segment; % flag to indicate first complete segment
        a_is_last_segment; % flag to indicate last complete segment
        a_last_data_mat; % matrix containing the last data window
        
        a_header_loaded; % flag to know if the header file was loaded or not
        a_nts_loaded;% flag to know if the note file was loaded or not
        
        a_n_bytes_ahead;
        a_abs_file_pt;
        
    end%End propreties
    
    properties (SetAccess = protected, GetAccess = protected)
        a_file_name;
        a_head_name;
        a_nts_name;
        a_file_elec_str; % string containing the names of all electrodes separated by a coma
        a_pat_id; % patient ID
        
        a_wsize_samp; % window size in samples
        a_step_samp; % step between data windows (in samples)
        
        a_mat_max_elem; % the maximum number of elements to be loaded in one single cycle
        a_chann_str; % string containing the names of the channels to process
        a_channs_ave_ind; % cell array containing the names of electrodes to be included in the average
        
    end%End propreties
    

    methods
        function self = bin_file(file_name,head_name,nts_name)
            %%
            %==============================================================
            % "bin_file" class constructor
            %
            % Input:
            %   file_name-->name of the binary file to be loaded
            %   head_name(optional)-->name of the header file. If this
            %       argument is empty or is not present, the header file will be
            %       constructed by replacing the last three letters of the binary
            %       file by the extention ".head"
            %
            % Output:
            %   self-->bin_file class object
            %
            %
            % EU FP7 Grant 211713 (EPILEPSIAE)
            %
            % Mario Valderrama
            % PARIS - CNRS
            % September 2009
            %==============================================================

            self.a_header_loaded = 0;
            
            self.a_nts_loaded = 0;
            
            if nargin < 1
                return;
            end
            if isempty(file_name)
                display('[bin_file] - ERROR: File name is empty!');
                return;
            end
            self.a_file_name = file_name;
            if ~exist(self.a_file_name, 'file')
                display('[bin_file] - ERROR: Binary file does not exist!');
                return;
            end
            
            if ~exist('head_name', 'var') || isempty(head_name)
                self.a_head_name = '';
            else
                self.a_head_name = head_name;
            end
            
            
            if ~exist(self.a_head_name, 'file')
                [str_pathstr, str_name] = fileparts(self.a_file_name);
                self.a_head_name = fullfile(str_pathstr, [str_name '.head']);
                if ~exist(self.a_head_name, 'file')
                    self.a_head_name = [self.a_file_name '.head'];
                    if ~exist(self.a_head_name, 'file')
                        display('[bin_file] - ERROR: Header file does not exist!');
                        return;
                    end
                end
            end
            self.get_header_info();
            
            
            
            if ~exist('nts_name', 'var') || isempty(head_name)
                self.a_nts_name = '';
            else
                self.a_nts_name = nts_name;
            end
            
            
            fprintf('nts name=%s\n',self.a_nts_name); %DG
            if ~exist(self.a_nts_name, 'file')
                [str_pathstr, str_name] = fileparts(self.a_file_name);
                self.a_nts_name = fullfile(str_pathstr, [str_name '.nts']);
                if ~exist(self.a_nts_name, 'file')
                    self.a_nts_name = [self.a_file_name '.nts'];
                    % DG commented this out. It does not appear to be a
                    % problem
%                     if ~exist(self.a_nts_name, 'file')
%                         display('[bin_file] - Warning: NTS file does not exist!')
%                     end
                end
            end
            
            
            
            if isempty(self.a_n_chan) || self.a_n_chan <= 0
                display('[bin_file] - ERROR: No channels in the file!');
                return;
            end
            
            if isempty(self.a_samp_freq) || self.a_samp_freq <= 0
                display('[bin_file] - ERROR: Sample frequency not present or wrong value!');
                return;
            end            
            
            %self.a_n_bytes = 2;
            
            sam = dir(self.a_file_name);
            sam = sam.bytes / ( self.a_n_bytes*self.a_n_chan);
            if self.a_n_samples == -1
                self.a_n_samples = sam;
            end
            if sam ~= self.a_n_samples
                display('[bin_file] - ERROR: The number of samples does not coincide between the header and the binary files!');
                return;
            end
            self.a_n_data_secs = self.a_n_samples / self.a_samp_freq;
            
            self.a_stop_ts = datestr(datenum(self.a_start_ts, 'yyyy-mm-dd HH:MM:SS') + ...
                datenum(0, 0, 0, 0, 0, self.a_n_data_secs), 'yyyy-mm-dd HH:MM:SS');
            
            self.a_duration_ts = datestr(datenum([0 0 0 0 0 0]) + ...
                datenum(0, 0, 0, 0, 0, self.a_n_data_secs), 'dd HH:MM:SS');
            
            if isempty(self.a_file_elec_str)
                for counter = 1:self.a_n_chan
                    if ~isempty(self.a_file_elec_str)
                        self.a_file_elec_str = [self.a_file_elec_str ','];
                    end
                    self.a_file_elec_str = [self.a_file_elec_str num2str(counter)];
                end
            end
            self.a_file_elec_cell = f_GetSignalNamesArray(self.a_file_elec_str);
            
            self.a_note_struct = [];

            
            self.a_mat_max_elem = 100 * 10^6;
            
            self.a_header_loaded = 1;
            
            self.a_abs_file_pt=self.a_first_data_bytes_offset+(self.a_n_samples*self.a_n_bytes*self.a_n_chan);
            %================Set data access flags==================
            self.a_is_last_segment=0;
            self.a_is_first_segment=1;
            self.a_curr_bytes_offset=self.a_first_data_bytes_offset;%Define the current data offset equal to the offset of the first data sample
            self.a_curr_samp_offset=0;

        end%End Constructor


        function get_header_info(self)
            %%
            %==============================================================
            % This functions reads the parameters stored in the header file
            %
            % Input:
            %
            % Output:
            %
            %
            % EU FP7 Grant 211713 (EPILEPSIAE)
            %
            % Mario Valderrama
            % PARIS - CNRS
            % September 2009
            %==============================================================              

            s_File = fopen(self.a_head_name, 'rt');
            if s_File == -1
                display(['[bin_file] - ERROR opening file: %s' self.a_head_name])
                return;
            end
            
            while 1
                str_Line = fgetl(s_File);
                if ~ischar(str_Line)
                    break;
                end
                [str_Token, str_Remain] = strtok(str_Line, '=');
                str_Token = strtrim(str_Token);
                if ~isempty(str_Remain) && length(str_Remain) >= 2
                    str_Remain = str_Remain(2:end);
                    str_Remain = strtrim(str_Remain);
                else
                    continue;
                end
                if strcmpi(str_Token, 'START_TS')
                    self.a_start_ts = str_Remain;
                    self.a_start_ts=datestr(datenum(self.a_start_ts,'yyyy-mm-dd HH:MM:SS'),'yyyy-mm-dd HH:MM:SS');
                end
                if strcmpi(str_Token, 'NUM_SAMPLES')
                    self.a_n_samples = str2num(str_Remain);
                end
                if strcmpi(str_Token, 'SAMPLE_FREQ')
                    self.a_samp_freq = str2num(str_Remain);
                end
                if strcmpi(str_Token, 'CONVERSION_FACTOR')
                    self.a_conv_factor = str2num(str_Remain);
                end
                if strcmpi(str_Token, 'NUM_CHANNELS')
                    self.a_n_chan = str2num(str_Remain);
                end
                if strcmpi(str_Token, 'ELEC_NAMES')
                    self.a_file_elec_str = str_Remain;
                    if self.a_file_elec_str(1) == '['
                        self.a_file_elec_str = self.a_file_elec_str(2:end);
                    end
                    if self.a_file_elec_str(end) == ']'
                        self.a_file_elec_str = self.a_file_elec_str(1:end - 1);
                    end
                end
                if strcmpi(str_Token, 'PAT_ID')
                    self.a_pat_id = str2num(str_Remain);
                end
                if strcmpi(str_Token, 'SAMPLE_BYTES')
                    self.a_n_bytes = str2num(str_Remain);
                end
                
            end
            fclose(s_File);
        end
        
        function get_notes(self)
            %==============================================================
            %"get_notes" method
            %This method acquires all the information related with the anotations
            %This information is:
            %   -sample where the event occurs
            %   -text describing the associated event
            %
            %This information is stored in the "note_struct" attribute that
            %is a struct array with fields 'sample' and 'note'
            %
            %
            %EU FP7 Grant 211713 (EPILEPSIAE)
            %
            %CÃ©sar A. D. Teixeira
            %CISUC, FCTUC, University of Coimbra
            %September 2009
            %==============================================================
            
            s_File = fopen(self.a_nts_name, 'rt');
            if s_File == -1
                display(['[bin_file] - ERROR opening file: %s' self.a_nts_name])
                return;
            end
            
            while 1
                str_Line = fgetl(s_File);
                
                d_idx=findstr(str_Line,':');
                sample=str2num(str_Line(1:d_idx(1)-1));
                if sample==0
                    break
                end
                note=str_Line(d_idx(1)+1:end);
                ixc = isstrprop(note, 'graphic');%Look for non-graphic characters at the end of the note string and eliminate them
                    ixc=find(ixc);
                    if size(ixc,2)~=0
                        ixc=ixc(end);
                        note=note(1:ixc);
                    end
                    self.a_note_struct=[self.a_note_struct,struct('sample',sample,'note',note)];%Pack sample number and note in
                    %a struct and append it to the 'a_note_struct' array
                
            end
            
            
            %self.a_note_struct=[];
        end%End get_notes method        
        
        function set_elect_ave(self, ave_elec_cell)
            %%
            %==============================================================
            % This functions sets the indices of electrodes to be
            % considered for an optional common average montage
            %
            % Input:
            %   ave_elec_cell-->cell array containing the names of
            %   electrodes to be included in the average
            %
            % Output:
            %
            %
            % EU FP7 Grant 211713 (EPILEPSIAE)
            %
            % Mario Valderrama
            % PARIS - CNRS
            % September 2009
            %==============================================================      
            
            self.a_channs_ave_cell = ave_elec_cell;
            self.a_channs_ave_ind = zeros(1, length(self.a_file_elec_cell));
            counter_sig = 0;
            for counter = 1:length(self.a_file_elec_cell)
                for s_Counter1 = 1:length(self.a_channs_ave_cell)
                    if ~strcmpi(self.a_file_elec_cell{counter}, self.a_channs_ave_cell{s_Counter1})
                        continue;
                    end
                    counter_sig = counter_sig + 1;
                    self.a_channs_ave_ind(counter_sig) = counter;
                    break;
                end
            end
            self.a_channs_ave_ind = self.a_channs_ave_ind(1:counter_sig);
        end
        
        
        function sigs_mat = get_bin_signals(self, first_sam, last_sam)
            %%
            %==============================================================
            % This functions reads the data from the binary files
            %
            % Input:
            %   first_sam(optional)-->number of the first sample to read from
            %       Default: first sample in the file
            %   last_sam(optional)-->number of the last sample to read to.
            %       Default: end of the file
            %
            % Output:
            %
            %
            % EU FP7 Grant 211713 (EPILEPSIAE)
            %
            % Mario Valderrama
            % PARIS - CNRS
            % September 2009
            %==============================================================                

            sigs_mat = [];
            
            if isempty(self.a_file_elec_cell) || isempty(self.a_channs_cell)
                return;
            end

            temp_EEG_file_name = '~eeg~temp~bin~mat.tmp';
            
            max_elem = self.a_mat_max_elem / 2;
            max_elem = floor(max_elem / self.a_n_chan) * self.a_n_chan;

            if isempty(first_sam)
                first_ind = 1;
            else
                first_ind = (first_sam - 1) * self.a_n_chan + 1;
            end
            cycles = 0;
            while 1
                cycles = cycles + 1;
                if isempty(last_sam)
                    last_ind = first_ind + max_elem - 1;
                else
                    last_ind = last_sam * self.a_n_chan;
                    if (last_ind - first_ind) + 1 > max_elem
                        last_ind = first_ind + max_elem - 1;
                    end
                end
                clear file_sig
                if self.a_n_bytes==2
                    file_sig = f_LoadI16File(self.a_file_name, first_ind, last_ind);
                else
                    file_sig = f_LoadI32File(self.a_file_name, first_ind, last_ind);
                end
                first_ind = last_ind + 1;
                file_sig = reshape(file_sig, self.a_n_chan, []);
                file_sig = file_sig.* self.a_conv_factor;

                clear ave_sig
                if ~isempty(self.a_channs_ave_ind)
                    ave_sig = mean(file_sig(self.a_channs_ave_ind, :));
                else
                    ave_sig = [];
                end
                
                clear sigs_mat
                sigs_mat = zeros(length(self.a_channs_cell), size(file_sig, 2));
                for counter = 1:length(self.a_channs_cell)
                    [sig1 sig2] = strtok(self.a_channs_cell{counter}, '-');
                    sig_ind = 0;
                    for counter1 = 1:length(self.a_file_elec_cell)
                        if ~strcmpi(self.a_file_elec_cell{counter1}, sig1)
                            continue;
                        end
                        sig_ind = counter1;
                        break;
                    end

                    if sig_ind <= 0
                        continue;
                    end

                    sigs_mat(counter, :) = file_sig(sig_ind, :);
                    if isempty(sig2)
                        if ~isempty(ave_sig)
                            sigs_mat(counter, :) = sigs_mat(counter, :) - ave_sig;
                        end
                    else
                        sig2 = sig2(2:end);
                        sig_ind = 0;
                        for counter1 = 1:length(self.a_file_elec_cell)
                            if ~strcmpi(self.a_file_elec_cell{counter1}, sig2)
                                continue;
                            end
                            sig_ind = counter1;
                            break;
                        end

                        if sig_ind <= 0
                            sigs_mat(counter, :) = zeros(1, size(file_sig, 2));
                            continue;
                        end

                        sigs_mat(counter, :) = sigs_mat(counter, :) - ...
                            file_sig(sig_ind, :);
                    end
                end

                if numel(file_sig) < max_elem && cycles <= 1
                    break;
                end

                append = 0;
                if cycles > 1
                    append = 1;
                end
                if ~isempty(sigs_mat)
                    f_SaveF64File(sigs_mat(:), temp_EEG_file_name, [], [], append);
                end
                clear sigs_mat

                if numel(file_sig) < max_elem
                    break;
                end
            end

            clear file_sig ave_sig

            if cycles > 1
                clear sigs_mat m_ECGSig m_RespSig
                sigs_mat = f_LoadF64File(temp_EEG_file_name);
                delete(temp_EEG_file_name);
                sigs_mat = reshape(sigs_mat, length(self.a_channs_cell), []);
            end
        end

        
        function [data,time] = def_data_access(self, wsize, step, channs_cell, offset)
            %%
            %==============================================================
            % "def_data_access" method
            % This method defines the way as EEG/ECG data will be accessed.
            % Parameters like data window size, time step between data
            % windows and the considered channels are defined by this
            % method.
            %
            % Inputs:
            %   wsize--> Window size in seconds
            %   step--> Time in seconds between windows. A step greater or
            %       equal to the window size result in non-overlaped
            %       windows. If step is not present or is empty, the step
            %       will be equal to wsize.
            %   offset--> Offset in samples from the beginning of the file
            %   channs_cell--> Cell array containing the name of the channels
            %       to be processed. The name can contain channels in
            %       MONOPOLAR, BIPOLAR or AVERAGE montage. A combination from
            %       some of them is also possible.
            %       Example:
            %           {'FP1';'FP2';'FPZ-FZ'}
            %            |__________|_______|
            %              Mono  Bipolar --> 2 channels Monopolar, 1 channel Bipolar
            %       For AVERAGE montage, a cell array containing the name of
            %       the channels to be included in the average must be
            %       declared before through the method set_elect_ave
            %
            %       If the names of the channels are not present in the
            %       header file, this argument (channs_cell) can contain a
            %       string with the indices of channels to be processed in
            %       any of the montages described above.
            %       For example:
            %           {'1';'3';'5';'6-7'}
            %
            % Outputs:
            %   res--> 1 if success; 0 otherwise
            %
            %
            % EU FP7 Grant 211713 (EPILEPSIAE)
            %
            % Mario Valderrama
            % PARIS - CNRS
            % September 2009
            %==============================================================
            
            if ~self.a_header_loaded
                display('[bin_file] - ERROR: header is not loaded!')
                return;
            end
            
            if ~exist('wsize', 'var') || isempty(wsize)
                return;
            end
            
            self.a_wsize = wsize;
            if ~exist('step', 'var') || isempty(step)
                self.a_step = wsize;
            else
                self.a_step = step;
            end       
            if ~exist('offset', 'var') || isempty(offset)
                self.a_first_data_samp_offset = 0;
            else
                self.a_first_data_samp_offset = offset;
            end
            if ~exist('channs_cell', 'var') || isempty(channs_cell)
                self.a_channs_cell = self.a_file_elec_cell;
            else
                self.a_channs_cell = channs_cell;
            end        
            
            self.a_wsize_samp = round(self.a_wsize * self.a_samp_freq);
            self.a_step_samp = round(self.a_step * self.a_samp_freq);
            
            self.a_n_bytes_ahead=self.a_step_samp*self.a_n_chan*self.a_n_bytes;%Compute the step in bytes
            
            %self.a_first_data_bytes_offset = self.a_first_data_samp_offset * ...
                self.a_n_bytes * self.a_n_chan;
            self.a_curr_bytes_offset = 0;
            self.a_last_bytes_offset = 0;
            self.a_curr_samp_offset = -1 * (self.a_step_samp - 1) + ...
                self.a_first_data_samp_offset;
            self.a_last_samp_offset = 0;
            
            self.a_is_first_segment = 1;
            self.a_is_last_segment = 0;
            self.a_last_data_mat = [];
            
            data = get_next_window(self);
            
            curr_sample=(self.a_curr_bytes_offset-self.a_first_data_bytes_offset)/self.a_n_bytes/self.a_n_chan;
            time=[curr_sample:curr_sample+size(data,2)-1].*(1/self.a_samp_freq);
            
            if self.a_curr_samp_offset==0
                    self.a_is_first_segment=1;
                else
                    self.a_is_first_segment=0;
                end
            
        end%End def_data_access method
        
        function data = redefine_data_access(self, wsize, step, offset)
            %%
            %==============================================================
            % "redefine_data_access" method
            % See def_data_access method.
            %
            % EU FP7 Grant 211713 (EPILEPSIAE)
            %
            % Mario Valderrama
            % PARIS - CNRS
            % September 2009
            %==============================================================
            
            data = self.def_data_access(wsize, step, self.a_channs_cell, offset);
        end        

        function data = get_next_window(self, back_dir)
            %%
            %==============================================================
            % "get_next_window" method
            % This method returns the next data segment, according to the
            % parameters defined in the "def_data_access" method.
            % Each call to this method advance the current data offset to a
            % new position, related with the defined step. If the end of
            % file was reached, in the next call a empty matrix is
            % returned.
            % The fisrt call to this method returns the first window.
            %
            %
            % EU FP7 Grant 211713 (EPILEPSIAE)
            %
            % Mario Valderrama
            % PARIS - CNRS
            % September 2009
            %==============================================================
            
            data = [];
            
            s_step_temp = self.a_step_samp;
            if exist('back_dir', 'var') && back_dir
                s_step_temp = -1 * s_step_temp;
            end
            
            
            
            s_first_data_samp = self.a_curr_samp_offset + s_step_temp;
            s_last_data_samp = s_first_data_samp + self.a_wsize_samp - 1;
            
             if s_first_data_samp <1 
                 data = self.a_last_data_mat;
                 self.a_is_first_segment=1;
                 return;
             end
            
            if s_first_data_samp >= self.a_n_samples
                self.a_is_last_segment=1;
                return;
            end
            
            if isempty(self.a_last_data_mat) || ...
                    s_first_data_samp > self.a_last_samp_offset || ...
                    s_last_data_samp < self.a_curr_samp_offset
                self.a_last_data_mat = self.get_bin_signals(s_first_data_samp, s_last_data_samp);
            else
                s_first_aux = [];
                
                if s_first_data_samp >= self.a_curr_samp_offset && ...
                        s_last_data_samp > self.a_last_samp_offset
                    
                    s_first_temp = self.a_last_samp_offset + 1;
                    s_last_temp = s_last_data_samp;
                    s_first_aux = s_first_data_samp - self.a_curr_samp_offset + 1;
                    s_last_aux = size(self.a_last_data_mat, 2);
                    
                    m_data_temp = self.get_bin_signals(s_first_temp, s_last_temp);
                    
                    self.a_last_data_mat = self.a_last_data_mat(:, s_first_aux:s_last_aux);
                    self.a_last_data_mat = f_AddHorElems(self.a_last_data_mat, m_data_temp);
                    clear m_data_temp;
                    
                elseif s_first_data_samp < self.a_curr_samp_offset && ...
                        s_last_data_samp <= self.a_last_samp_offset
                    
                    s_first_temp = s_first_data_samp;
                    s_last_temp = self.a_curr_samp_offset - 1;
                    s_first_aux = 1;
                    s_last_aux = s_last_data_samp - self.a_curr_samp_offset + 1;
                    
                    m_data_temp = self.get_bin_signals(s_first_temp, s_last_temp);
                    
                    m_data_temp = f_AddHorElems(m_data_temp, ...
                        self.a_last_data_mat(:, s_first_aux:s_last_aux));
                    self.a_last_data_mat = m_data_temp;
                    
                    
                    
                    clear m_data_temp;
                    
                elseif s_first_data_samp >= self.a_curr_samp_offset && ...
                        s_last_data_samp <= self.a_last_samp_offset
                    
                    s_first_aux = s_first_data_samp - self.a_curr_samp_offset + 1;
                    s_last_aux = s_last_data_samp - self.a_curr_samp_offset + 1;
                    
                    self.a_last_data_mat = self.a_last_data_mat(:, s_first_aux:s_last_aux);
                    
                end       
                
                if isempty(s_first_aux)
                    return;
                end
                
                
                
            end
            
            self.a_curr_samp_offset = s_first_data_samp;
            self.a_last_samp_offset = s_last_data_samp;
            self.a_curr_bytes_offset = (self.a_curr_samp_offset - 1) * self.a_n_bytes * ...
                self.a_n_chan;
            self.a_last_bytes_offset = (self.a_last_samp_offset - 1) * self.a_n_bytes * ...
                self.a_n_chan;
            
            if self.a_curr_samp_offset + self.a_step_samp + self.a_wsize_samp - 1 >= ...
                    self.a_n_samples
                self.a_is_last_segment = 1;
            else
                self.a_is_last_segment = 0;
            end
            
            if self.a_curr_samp_offset > self.a_first_data_samp_offset
                self.a_is_first_segment = 0;
            else
                self.a_is_first_segment = 1;
            end     
            
            data = self.a_last_data_mat;
           
            
        end%End get_next_window method
        
        function data = get_prev_window(self)
            %%
            %==============================================================
            % "get_prev_window" method
            % This method returns the previous data segment, according to the
            % parameters defined in the "def_data_access" method.
            %
            %
            % EU FP7 Grant 211713 (EPILEPSIAE)
            %
            % Mario Valderrama
            % PARIS - CNRS
            % September 2009
            %==============================================================
            
            data = self.get_next_window(1);
            
        end%End get_next_window method

    end%End Methods
end%End Class
