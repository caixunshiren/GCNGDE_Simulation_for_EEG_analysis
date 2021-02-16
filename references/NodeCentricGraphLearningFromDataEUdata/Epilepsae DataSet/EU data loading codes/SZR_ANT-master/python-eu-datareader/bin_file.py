# -*- coding: utf-8 -*-
import os
import datetime
import numpy as np
from f_LoadI16File import f_LoadI16File
from f_LoadI32File import f_LoadI32File

class bin_file: # matlab version inherits handle superclass which performs all assignments by reference; i.e. a = b; a.number = 7; => b.number = 7
    """
    Python class that defines a "bin_file" object.
       
    This class is intended to load and read data stored in raw binary
    files according to the following specifications:
    
    1. Each file consists actually in a couple of files: the first one
    contains the raw data stored in binary values (binary file); the
    second one corresponds to a text file containing some parameters
    about how the data was acquired and stored (header file).
            
    2. The parameters present in the header file (as it was decided in
    the meeting in Coimbra on July 2009) are the following:
    
    start_ts=<yyyy-mm-dd hh:mm:ss.microseconds>
    num_samples=<int>
    sample_freq=<int>
    conversion_factor=<float>
    num_channels:=<int>
    (optional)elec_names=<string>, ...  (e.g. FP1,FP2)
    (opt.)pat_id=<int>
    
    3. The binary file is organized as follows:
    
    Each sample (data value) in the file is stored in 2 bytes, signed.
    (“short” type in C, “int16” in MATLAB)
     The firsts 2 bytes in the file correspond to the value of the first
    sample for the first channel, the next 2 to the value of the first
    sample for the second channel and so on until the first sample for
    the last channel (according to parameter “num_channels” above). After
    the first sample for all channels, the next series will be the second
    sample for each progressive channel and the sequence continues until
    the end of the file.
    """

#===============================================================================
    def __init__(self,file_name = '',head_name = '',nts_name = ''):
        """
        bin_file class constructor
            
        Input:
          file_name-->name of the binary file to be loaded
          head_name(optional)-->name of the header file. If this
              argument is empty or is not present, the header file will be
              constructed by replacing the last three letters of the binary
              file by the extention ".head"
        
        Output:
          self-->bin_file class object
        
        
        EU FP7 Grant 211713 (EPILEPSIAE)
        
        Mario Valderrama
        PARIS - CNRS
        September 2009
        """
        
        # initialize a few attributes
        self.a_header_loaded = 0
        self.a_nts_loaded = 0
        self.a_n_chan = -1
        self.a_samp_freq = -1
        self.a_n_samples = -1
        self.a_first_data_bytes_offset = 0
        self.a_file_elec_cell = []
        self.a_channs_cell = []
        self.a_channs_ave_ind = []
        
        # input checking
        if file_name == '':
            raise Exception('[bin_file] - ERROR: File name is empty!')
        self.a_file_name = file_name # check if actually valid later
        self.a_head_name = head_name # check if actually valid later

        self.get_header_info()
        
        self.a_nts_name = nts_name # optional argument
        if self.a_nts_name == '':
            print('WARNING: NTS file does not exist')
        
        if self.a_n_chan < 0:
            raise Exception('[bin_file] - ERROR: No channels in the file!')
        
        if self.a_samp_freq < 0:
            raise Exception('[bin_file] - ERROR: Sample frequency not present or wrong value!')
            
        sam = os.path.getsize(self.a_file_name) # get size of file in bytes
        sam = sam / (self.a_n_bytes * self.a_n_chan)
        if self.a_n_samples == -1:
            self.a_n_samples = sam
        if sam != self.a_n_samples:
            raise Exception('[bin_file] - ERROR: The number of samples does not coincide between the header and the binary files!')
        
        # start time, end time duration with proper formatting
        self.a_n_data_secs = self.a_n_samples / self.a_samp_freq
        start_date = datetime.datetime.strptime(self.a_start_ts,'%Y-%m-%d %H:%M:%S.%f')
        end_date = start_date + datetime.timedelta(seconds = self.a_n_data_secs)
        self.a_start_ts = start_date.strftime('%d-%b-%Y %H:%M:%S')
        self.a_stop_ts = end_date.strftime('%d-%b-%Y %H:%M:%S')
        dur = datetime.datetime.strptime('00','%H')
        self.a_duration_ts = dur + datetime.timedelta(seconds = self.a_n_data_secs)
        self.a_duration_ts = self.a_duration_ts.strftime('%H:%M:%S')
        
        self.a_file_elec_cell = self.a_file_elec_str.split(',')
        
        self.a_note_struct = []
        
        self.a_mat_max_elem = 100 * 10**6
        
        self.a_header_loaded = 1
        
        self.a_abs_file_pt = self.a_first_data_bytes_offset + (self.a_n_samples * self.a_n_bytes * self.a_n_chan)
        
        # set data access flags
        self.a_is_last_segment = 0
        self.a_is_first_segment = 1
        self.a_curr_bytes_offset = self.a_first_data_bytes_offset # define the current data offset equal to the offset of the first data sample
        self.a_curr_samp_offset = 0;
        
#===============================================================================
    def get_header_info(self):
        """
	  	This functions reads the parameters stored in the header file
        
        Input:
        
        Output:
        
        
        EU FP7 Grant 211713 (EPILEPSIAE)
        
        Mario Valderrama
        PARIS - CNRS
        September 2009
        """

        try:
        	s_file = open(self.a_head_name,'r')
        except:
        	raise Exception('[bin_file] - ERROR opening ' + self.a_head_name)

        while True:
            str_Line = s_file.readline()
            str_Line = str_Line.strip()
            if not str_Line or not isinstance(str_Line,basestring):
                break
            s = str_Line.split("=",1)
            str_Token = s[0].strip()
            str_Token = str_Token.upper()
            str_Remain = s[1].strip()
            if str_Token == 'START_TS':
                self.a_start_ts = str_Remain
                # FORMAT date nicely here // self.a_start_ts = 
            elif str_Token == 'SAMPLE_FREQ':
                self.a_samp_freq = float(str_Remain)
            elif str_Token == 'CONVERSION_FACTOR':
                self.a_conv_factor = float(str_Remain)
            elif str_Token == 'NUM_CHANNELS':
                self.a_n_chan = float(str_Remain)
            elif str_Token == 'ELEC_NAMES':
                self.a_file_elec_str = str_Remain
                end = len(self.a_file_elec_str)
                if self.a_file_elec_str[0] == '[':
                    self.a_file_elec_str = self.a_file_elec_str[1:end]
                end = len(self.a_file_elec_str)
                if self.a_file_elec_str[end - 1] == ']':
                    self.a_file_elec_str = self.a_file_elec_str[0:end - 1]
            elif str_Token == 'PAT_ID':
                self.a_pat_id = float(str_Remain)
            elif str_Token == 'SAMPLE_BYTES':
                self.a_n_bytes = float(str_Remain)

        s_file.close()

#===============================================================================
    def get_notes(self):
        """
        TO BE IMPLEMENTED: NO SAMPLE NTS FILE TO USE
        """    
        
        pass
        
    
#===============================================================================
    def set_elect_ave(self,ave_elec_cell):
        """
        This function sets the indices of electrodes to be considered for an optional common average montage
        
        Input:
            ave_elec_cell->python list containing the names of electrodes to be included in the average
            
        Output:
            none (but object attributes set)
        """
        
        self.a_channs_ave_cell = ave_elec_cell
        self.a_channs_ave_ind = []
        counter_sig = 0
        for counter in range(0,len(self.a_file_elec_cell)):
            for s_Counter1 in range(0,len(self.a_channs_ave_cell)):
                if self.a_file_elec_cell[counter] != self.a_channs_ave_cell[s_Counter1]:
                    continue
                counter_sig += 1
                self.a_channs_ave_ind.append(counter)
                break
        self.a_channs_ave_ind = self.a_channs_ave_ind[0:counter_sig]
    
#===============================================================================
    def get_bin_signals(self,first_sam = -1,last_sam = -1):
        """
        This fucntion reads the data from the binary files
        
        Input:
            first_sam(optional)-->number of the first sample to read from
                Default: first sample in the file
            last_sam(optional)-->number of the last sample to read to
                Default: end of the file
        Output:
            sigs_mat
        """
        
        if len(self.a_file_elec_cell) == 0 or len(self.a_channs_cell) == 0:
            return
            
        temp_EEG_file_name = '~eeg~temp~bin~mat.tmp' # ??
        
        max_elem = self.a_mat_max_elem / 2
        max_elem = ((max_elem / self.a_n_chan) // 1) * self.a_n_chan
        
        if first_sam == -1:
            first_ind = 1
        else:
            first_ind = (first_sam - 1) * self.a_n_chan + 1
        
        cycles = 0
        while 1:
            cycles += 1
            if last_sam == -1:
                last_ind = first_ind + max_elem - 1
            else:
                last_ind = last_sam * self.a_n_chan
                if (last_ind - first_ind) + 1 > max_elem:
                    last_ind = first_ind + max_elem - 1
            
            first_ind = int(first_ind)
            last_ind = int(last_ind)
            if self.a_n_bytes == 2:
                file_sig = f_LoadI16File(self.a_file_name,first_ind,last_ind)
            else:
            	file_sig = f_LoadI32File(self.a_file_name,first_ind,last_ind)
            first_ind = last_ind + 1
            file_sig = np.reshape(file_sig,(self.a_n_chan,-1),'F')
            file_sig = file_sig * self.a_conv_factor
    
            if self.a_channs_ave_ind != []:
            	ave_sig = np.ndarray.mean(file_sig[self.a_channs_ave_ind - 1,:])
            else:
            	ave_sig = []
    
            sigs_mat = np.zeros((len(self.a_channs_cell),file_sig.shape[1]))
            for counter in range(0,len(self.a_channs_cell)):
            	sig = self.a_channs_cell[counter].split('-')
            	sig1 = sig[0]
            	sig2 = ''.join(sig[1:len(sig)])
            	sig_ind = -1
            	for counter1 in range(0,len(self.a_file_elec_cell)):
            		if self.a_file_elec_cell[counter1] != sig1:
            			continue
            		sig_ind = counter1
            		break
    
            	if sig_ind < 0:
            		continue
    
            	sigs_mat[counter,:] = file_sig[sig_ind,:]
            	if len(sig2) == 0:
                     if ave_sig != []:
                         sigs_mat[counter,:] = sigs_mat[counter,:] - ave_sig
            	else:
                     sig2 = sig2[1:len(sig2)]
                     sig_ind = -1
                     for counter1 in range(0,len(self.a_file_elec_cell)):
                         if self.a_file_elec_cell[counter1] != sig2:
                             continue
                         sig_ind = counter1
                         break
                     if sig_ind < 0:
                         sigs_mat[counter,:] = np.zeros((1,file_sig.shape[1]))
                         continue
                     sigs_mat[counter,:] = sigs_mat[counter,:] - file_sig[sig_ind,:]
            if file_sig.size < max_elem and cycles <= 1:
                break
            
            # ... more to implement here but don't have access to functions
            raise Exception('This section of get_bin_signals has not been implemented')            
            
        return sigs_mat    
            
            
#===============================================================================
    def def_data_access(self,wsize = -1,step = -1,channs_cell = [],offset = -1):
        """
        """

        if self.a_header_loaded == 0:
            raise Exception('[bin_file] - ERROR: header is not loaded!')
            
        if wsize == -1:
            return
        self.a_wsize = wsize
        
        if step == -1:
            self.a_step = wsize
        else:
            self.a_step = step
        
        if offset == -1:
            self.a_first_data_samp_offset = 0
        else:
            self.a_first_data_samp_offset = offset
            
        if channs_cell == []:
            self.a_channs_cell = self.a_file_elec_cell
        else:
            self.a_channs_cell = channs_cell
            
        self.a_wsize_samp = round(self.a_wsize * self.a_samp_freq)
        self.a_step_samp = round(self.a_step * self.a_samp_freq)
        
        self.a_n_bytes_ahead = self.a_step_samp * self.a_n_chan * self.a_n_bytes # computer the step in bytes
        
        self.a_curr_bytes_offset = 0
        self.a_last_bytes_offset = 0
        self.a_curr_samp_offset = -1 * (self.a_step_samp - 1) + self.a_first_data_samp_offset
        self.a_last_samp_offset = 0
        
        self.a_is_first_segment = 1
        self.a_is_last_segment = 0
        self.a_last_data_mat = []
        
        data = self.get_next_window()
        
        curr_sample = (self.a_curr_bytes_offset - self.a_first_data_bytes_offset) / self.a_n_bytes / self.a_n_chan
        time = np.arange(curr_sample,curr_sample + data.shape[1]) * (1 / self.a_samp_freq)
        
        if self.a_curr_samp_offset == 0:
            self.a_is_first_segment = 1
        else:
            self.a_is_first_segment = 0
        
        return (data,time)
    
#===============================================================================
    def redefine_data_access(self,wsize,step,offset):
        """
        """
        
        data = self.def_data_access(wsize,step,self.a_channs_cell,offset)
        return data
    
#===============================================================================
    def get_next_window(self,back_dir = 0):
        """
        Returns the next data segment, according to the parameters defined in the def_data_access method. 
        Each call to this method advances the current data offset to a new position, related with the defined step.
        If the end of the file was reached, in the next call an empty matrix is returned.
        The first call to this method returns the first window.
        
        Input:
            back_dir--> direction (0 = forwards, 1 = backwards)
        """
        
        data = []
        
        s_step_temp = self.a_step_samp
        if back_dir == 1:
            s_step_temp *= -1
            
        s_first_data_samp = self.a_curr_samp_offset + s_step_temp
        s_last_data_samp = s_first_data_samp + self.a_wsize_samp - 1
        
        if s_first_data_samp < 1:
            data = self.a_last_data_mat
            self.a_is_first_segment = 1
            return data
        
        if s_first_data_samp >= self.a_n_samples:
            self.a_is_last_segment = 1
            return data
            
        if self.a_last_data_mat == [] or s_first_data_samp > self.a_last_samp_offset or s_last_data_samp < self.a_curr_samp_offset:
            self.a_last_data_mat = self.get_bin_signals(s_first_data_samp,s_last_data_samp)
        else:
            s_first_aux = []
            if s_first_data_samp >= self.a_curr_samp_offset and s_last_data_samp > samp.a_last_samp_offset:
                raise Exception('ERROR: f_AddHorElems not implemented, aborting')
                s_first_temp = self.a_last_samp_offset + 1
                s_last_temp = s_last_data_samp
                s_first_aux = s_first_data_samp - self.a_curr_samp_offset + 1
                s_last_aux = self.a_last_data_mat.shape[1] # numpy array
                
                m_data_temp = self.get_bin_signals(s_first_temp,s_last_temp)
                
                self.a_last_data_mat = self.a_last_data_mat[:,s_first_aux - 1:s_last_aux - 1] # -1 necessary
                self.a_last_data_mat = f_AddHorElems(self.a_last_data_mat,m_data_temp) # f_AddHorElems doesn't exist
                
            elif s_first_data_samp < self.a_curr_samp_offset and s_last_data_samp <= self.a_last_samp_offset:
                raise Exception('ERROR: f_AddHorElems not implemented, aborting')
                s_first_temp = s_first_data_samp
                s_last_temp = self.a_curr_samp_offset - 1
                s_first_aux = 1
                s_last_aux = s_last_data_samp - self.a_curr_samp_offset + 1
                
                m_data_temp = self.get_bin_signals(s_first_temp,s_last_temp)
                
                m_data_temp = f_AddHorElems(m_data_temp,self.a_last_data_mat[:,s_first_aux - 1:s_last_aux - 1])
                self.a_last_data_mat = m_data_temp
            elif s_first_data_samp >= self.a_curr_samp_offset and s_last_data_samp <= self.a_last_samp_offset:
                s_first_aux = s_first_data_samp - self.a_curr_samp_offset + 1
                s_last_aux = s_last_data_samp - self.a_curr_samp_offset + 1
                
                self.a_last_data_mat = self.a_last_data_mat[:,s_first_aux - 1:s_last_aux - 1]
                
            if s_first_aux == []:
                return data
            
        self.a_curr_samp_offset = s_first_data_samp
        self.a_last_samp_offset = s_last_data_samp
        self.a_curr_bytes_offset = (self.a_curr_samp_offset - 1) * self.a_n_bytes * self.a_n_chan
        self.a_last_bytes_offset = (self.a_last_samp_offset - 1) * self.a_n_bytes * self.a_n_chan
        
        if self.a_curr_samp_offset + self.a_step_samp + self.a_wsize_samp - 1 >= self.a_n_samples:
            self.a_is_last_segment = 1
        else:
            self.a_is_last_segment = 0
        
        if self.a_curr_samp_offset > self.a_first_data_samp_offset:
            self.a_is_first_segment = 0
        else:
            self.a_is_first_segment = 1
        
        data = self.a_last_data_mat
        return data
        
#===============================================================================
    def get_prev_window(self):
        """
        Returns the previous data segment, according to the parameters defined in the def_data_access method.
        """
        
        data = self.get_next_window(1)
        return data
    
#===============================================================================
            