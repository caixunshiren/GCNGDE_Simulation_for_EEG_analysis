import os
import pyedflib
import numpy as np

def seizure_loadedf(comp_file_name):
    text_file = open(comp_file_name + '.seizures', "rb")
    byte_array = text_file.read()
    byte_array_chars = np.array(list(byte_array))
    number_of_seizures = int((np.sum(byte_array_chars == int('ec', 16) ) - 1) / 2)
    seizure_start_time_offsets = np.zeros((number_of_seizures,))
    seizure_lengths = np.zeros((number_of_seizures,))
    for i in np.arange(number_of_seizures)+1:
        seizure_start_time_offsets[i-1] = int( bin(byte_array_chars[22 + (i*16)])[2:] + bin(byte_array_chars[25 + (i*16)])[2:] , 2)
        seizure_lengths[i-1] = byte_array_chars[33 + (i*16)]
    text_file.close()
    return seizure_start_time_offsets, seizure_lengths


def inner_loadedf(input_prefix):
    f = pyedflib.EdfReader(input_prefix)
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    sigbufs = np.zeros((n, f.getNSamples()[0]))
    for i in np.arange(n):
        sigbufs[i, :] = f.readSignal(i)
    return sigbufs


def load_edf_data(input_prefix, target, load_Core):
    dir = input_prefix + '/' + target
    filenames = sorted(os.listdir(dir))
    counter_to_load = 0
    for i, filename in enumerate(filenames):
        if(i<load_Core.start_num):
            continue
        comp_file_name = dir + '/' + filename
        if(filename.find('.edf')!=-1 and filename.find('.seizures')==-1):
            data = inner_loadedf( comp_file_name)
                        
            if os.path.exists(comp_file_name + '.seizures'):
                seizure_start_time_offsets, seizure_lengths = seizure_loadedf( comp_file_name )
            else:
                if(load_Core.only_seizures):
                    continue
                else:
                    seizure_start_time_offsets = -1
                    seizure_lengths = -1
            print('    edf loading settings. Offset: %f, Length:s %f' % (seizure_start_time_offsets, seizure_lengths ))
            yield(data, filename.split('.')[0], seizure_start_time_offsets, seizure_lengths)
            if(load_Core.howmany_to_load != -1):
                if(counter_to_load == load_Core.howmany_to_load-1):
                    break
                else:
                    counter_to_load += 1
                    
                    