import numpy as np
from scipy import signal
from scipy.signal import resample, hann
from sklearn import preprocessing
import tensorflow as tf
from scipy.signal import butter, lfilter
import networkx as nx
# optional modules for trying out different transforms
# try:
#     import pywt
# except ImportError, e:
#     pass
# 
# try:
#     from scikits.talkbox.features import mfcc
# except ImportError, e:
#     pass


# NOTE(mike): All transforms take in data of the shape (NUM_CHANNELS, NUM_FEATURES)
# Although some have been written work on the last axis and may work on any-dimension data.


class FFT:
    """
    Apply Fast Fourier Transform to the last axis.
    """
    def get_name(self):
        return "fft"

    def apply(self, data, target):
        return tf.spectral.rfft(data)


class Slice:
    """
    Take a slice of the data on the last axis.
    e.g. Slice(1, 48) works like a normal python slice, that is 1-47 will be taken
    """
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def get_name(self):
        return "slice%d-%d" % (self.start, self.end)

    def apply(self, data, target):
        s = [slice(None),] * data.ndim
        s[-1] = slice(self.start, self.end)
        return data[s]


class LPF:
    """
    Low-pass filter using FIR window
    """
    def __init__(self, f):
        self.f = f

    def get_name(self):
        return 'lpf%d' % self.f

    def apply(self, data, target):
        nyq = self.f / 2.0
        cutoff = min(self.f, nyq-1)
        h = signal.firwin(numtaps=101, cutoff=cutoff, nyq=nyq)

        # data[i][ch][dim0]
        for i in range(len(data)):
            data_point = data[i]
            for j in range(len(data_point)):
                data_point[j] = signal.lfilter(h, 1.0, data_point[j])

        return data


# class MFCC:
#     """
#     Mel-frequency cepstrum coefficients
#     """
#     def get_name(self):
#         return "mfcc"
# 
#     def apply(self, data, target):
#         all_ceps = []
#         for ch in data:
#             ceps, mspec, spec = mfcc(ch)
#             all_ceps.append(ceps.ravel())
# 
#         return np.array(all_ceps)


class Magnitude:
    """
    Take magnitudes of Complex data
    """
    def get_name(self):
        return "mag"

    def apply(self, data, target):
        return np.absolute(data)


class MagnitudeAndPhase:
    """
    Take the magnitudes and phases of complex data and append them together.
    """
    def get_name(self):
        return "magphase"

    def apply(self, data, target):
        magnitudes = np.absolute(data)
        phases = np.angle(data)
        return np.concatenate((magnitudes, phases), axis=1)


class Log10:
    """
    Apply Log10
    """
    def get_name(self):
        return "log10"

    def apply(self, data, target):
        # 10.0 * log10(re * re + im * im)
        indices = np.where(data <= 0)
        data[indices] = np.max(data)
        data[indices] = (np.min(data) * 0.1)
        return np.log10(data)


class Stats:
    """
    Subtract the mean, then take (min, max, standard_deviation) for each channel.
    """
    def get_name(self):
        return "stats"

    def apply(self, data, target):
        # data[ch][dim]
        shape = data.shape
        out = np.empty((shape[0], 3))
        for i in range(len(data)):
            ch_data = data[i]
            ch_data = data[i] - np.mean(ch_data)
            outi = out[i]
            outi[0] = np.std(ch_data)
            outi[1] = np.min(ch_data)
            outi[2] = np.max(ch_data)

        return out


class Resample:
    """
    Resample time-series data.
    """
    def __init__(self, sample_rate):
        self.f = sample_rate

    def get_name(self):
        return "resample%d" % self.f

    def apply(self, data, target):
        axis = data.ndim - 1
        if data.shape[-1] > self.f:
            return resample(data, self.f, axis=axis)
        return data


class ResampleHanning:
    """
    Resample time-series data using a Hanning window
    """
    def __init__(self, sample_rate):
        self.f = sample_rate

    def get_name(self):
        return "resample%dhanning" % self.f

    def apply(self, data, target):
        axis = data.ndim - 1
        out = resample(data, self.f, axis=axis, window=hann(M=data.shape[axis]))
        return out


# class DaubWaveletStats:
#     """
#     Daubechies wavelet coefficients. For each block of co-efficients
#     take (mean, std, min, max)
#     """
#     def __init__(self, n):
#         self.n = n
# 
#     def get_name(self):
#         return "dwtdb%dstats" % self.n
# 
#     def apply(self, data, target):
#         # data[ch][dim0]
#         shape = data.shape
#         out = np.empty((shape[0], 4 * (self.n * 2 + 1)), dtype=np.float64)
# 
#         def set_stats(outi, x, offset):
#             outi[offset*4] = np.mean(x)
#             outi[offset*4+1] = np.std(x)
#             outi[offset*4+2] = np.min(x)
#             outi[offset*4+3] = np.max(x)
# 
#         for i in range(len(data)):
#             outi = out[i]
#             new_data = pywt.wavedec(data[i], 'db%d' % self.n, level=self.n*2)
#             for i, x in enumerate(new_data):
#                 set_stats(outi, x, i)
# 
#         return out


class UnitScale:
    """
    Scale across the last axis.
    """
    def get_name(self):
        return 'unit-scale'

    def apply(self, data, target):
        return preprocessing.scale(data, axis=data.ndim-1)


class UnitScaleFeat:
    """
    Scale across the first axis, i.e. scale each feature.
    """
    def get_name(self):
        return 'unit-scale-feat'

    def apply(self, data, target):
        return preprocessing.scale(data, axis=0)


class CorrelationMatrix:
    """
    Calculate correlation coefficients matrix across all EEG channels.
    """
    def get_name(self):
        return 'corr-mat'

    def apply(self, data, target):
        return np.corrcoef(data)


class Eigenvalues:
    """
    Take eigenvalues of a matrix, and sort them by magnitude in order to
    make them useful as features (as they have no inherent order).
    """
    def get_name(self):
        return 'eigenvalues'

    def apply(self, data, target):
        w, v = np.linalg.eig(data)
        w = np.absolute(w)
        w.sort()
        return w


# Take the upper right triangle of a matrix
def upper_right_triangle(matrix):
    accum = []
    for i in range(matrix.shape[0]):
        for j in range(i+1, matrix.shape[1]):
            accum.append(matrix[i, j])

    return np.array(accum)


class OverlappingFFTDeltas:
    """
    Calculate overlapping FFT windows. The time window will be split up into num_parts,
    and parts_per_window determines how many parts form an FFT segment.

    e.g. num_parts=4 and parts_per_windows=2 indicates 3 segments
    parts = [0, 1, 2, 3]
    segment0 = parts[0:1]
    segment1 = parts[1:2]
    segment2 = parts[2:3]

    Then the features used are (segment2-segment1, segment1-segment0)

    NOTE: Experimental, not sure if this works properly.
    """
    def __init__(self, num_parts, parts_per_window, start, end):
        self.num_parts = num_parts
        self.parts_per_window = parts_per_window
        self.start = start
        self.end = end

    def get_name(self):
        return "overlappingfftdeltas%d-%d-%d-%d" % (self.num_parts, self.parts_per_window, self.start, self.end)

    def apply(self, data, target):
        axis = data.ndim - 1

        parts = np.split(data, self.num_parts, axis=axis)

        #if slice end is 208, we want 208hz
        partial_size = (1.0 * self.parts_per_window) / self.num_parts
        #if slice end is 208, and partial_size is 0.5, then end should be 104
        partial_end = int(self.end * partial_size)

        partials = []
        for i in range(self.num_parts - self.parts_per_window + 1):
            combined_parts = parts[i:i+self.parts_per_window]
            if self.parts_per_window > 1:
                d = np.concatenate(combined_parts, axis=axis)
            else:
                d = combined_parts
            d = Slice(self.start, partial_end).apply(np.fft.rfft(d, axis=axis))
            d = Magnitude().apply(d)
            d = Log10().apply(d)
            partials.append(d)

        diffs = []
        for i in range(1, len(partials)):
            diffs.append(partials[i] - partials[i-1])

        return np.concatenate(diffs, axis=axis)


class FFTWithOverlappingFFTDeltas:
    """
    As above but appends the whole FFT to the overlapping data.

    NOTE: Experimental, not sure if this works properly.
    """
    def __init__(self, num_parts, parts_per_window, start, end):
        self.num_parts = num_parts
        self.parts_per_window = parts_per_window
        self.start = start
        self.end = end

    def get_name(self):
        return "fftwithoverlappingfftdeltas%d-%d-%d-%d" % (self.num_parts, self.parts_per_window, self.start, self.end)

    def apply(self, data, target):
        axis = data.ndim - 1

        full_fft = np.fft.rfft(data, axis=axis)
        full_fft = Magnitude().apply(full_fft)
        full_fft = Log10().apply(full_fft)

        parts = np.split(data, self.num_parts, axis=axis)

        #if slice end is 208, we want 208hz
        partial_size = (1.0 * self.parts_per_window) / self.num_parts
        #if slice end is 208, and partial_size is 0.5, then end should be 104
        partial_end = int(self.end * partial_size)

        partials = []
        for i in range(self.num_parts - self.parts_per_window + 1):
            d = np.concatenate(parts[i:i+self.parts_per_window], axis=axis)
            d = Slice(self.start, partial_end).apply(np.fft.rfft(d, axis=axis))
            d = Magnitude().apply(d)
            d = Log10().apply(d)
            partials.append(d)

        out = [full_fft]
        for i in range(1, len(partials)):
            out.append(partials[i] - partials[i-1])

        return np.concatenate(out, axis=axis)


class FreqCorrelation:
    """
    Correlation in the frequency domain. First take FFT with (start, end) slice options,
    then calculate correlation co-efficients on the FFT output, followed by calculating
    eigenvalues on the correlation co-efficients matrix.

    The output features are (fft, upper_right_diagonal(correlation_coefficients), eigenvalues)

    Features can be selected/omitted using the constructor arguments.
    """
    def __init__(self, start, end, scale_option, with_fft=False, with_corr=True, with_eigen=True):
        self.start = start
        self.end = end
        self.scale_option = scale_option
        self.with_fft = with_fft
        self.with_corr = with_corr
        self.with_eigen = with_eigen
        assert scale_option in ('us', 'usf', 'none')
        assert with_corr or with_eigen

    def get_name(self):
        selections = []
        if not self.with_corr:
            selections.append('nocorr')
        if not self.with_eigen:
            selections.append('noeig')
        if len(selections) > 0:
            selection_str = '-' + '-'.join(selections)
        else:
            selection_str = ''
        return 'freq-correlation-%d-%d-%s-%s%s' % (self.start, self.end, 'withfft' if self.with_fft else 'nofft',
                                                   self.scale_option, selection_str)

    def apply(self, data, target):
        data1 = FFT().apply(data, target)
        data1 = Slice(self.start, self.end).apply(data1, target)
        data1 = Magnitude().apply(data1, target)
        data1 = Log10().apply(data1, target)

        data2 = data1
        if self.scale_option == 'usf':
            data2 = UnitScaleFeat().apply(data2, target)
        elif self.scale_option == 'us':
            data2 = UnitScale().apply(data2, target)

        data2 = CorrelationMatrix().apply(data2, target)

        if self.with_eigen:
            w = Eigenvalues().apply(data2, target)

        out = []
        if self.with_corr:
            data2 = upper_right_triangle(data2)
            out.append(data2)
        if self.with_eigen:
            out.append(w)
        if self.with_fft:
            data1 = data1.ravel()
            out.append(data1)
        for d in out:
            assert d.ndim == 1

        return np.concatenate(out, axis=0)


class TimeCorrelation:
    """
    Correlation in the time domain. First downsample the data, then calculate correlation co-efficients
    followed by calculating eigenvalues on the correlation co-efficients matrix.

    The output features are (upper_right_diagonal(correlation_coefficients), eigenvalues)

    Features can be selected/omitted using the constructor arguments.
    """
    def __init__(self, max_hz, scale_option, with_corr=True, with_eigen=True):
        self.max_hz = max_hz
        self.scale_option = scale_option
        self.with_corr = with_corr
        self.with_eigen = with_eigen
        assert scale_option in ('us', 'usf', 'none')
        assert with_corr or with_eigen

    def get_name(self):
        selections = []
        if not self.with_corr:
            selections.append('nocorr')
        if not self.with_eigen:
            selections.append('noeig')
        if len(selections) > 0:
            selection_str = '-' + '-'.join(selections)
        else:
            selection_str = ''
        return 'time-correlation-r%d-%s%s' % (self.max_hz, self.scale_option, selection_str)

    def apply(self, data, target):
        # so that correlation matrix calculation doesn't crash
        for ch in data:
            if np.alltrue(ch == 0.0):
                ch[-1] += 0.00001

        data1 = data
        if data1.shape[1] > self.max_hz:
            data1 = Resample(self.max_hz).apply(data1, target)

        if self.scale_option == 'usf':
            data1 = UnitScaleFeat().apply(data1, target)
        elif self.scale_option == 'us':
            data1 = UnitScale().apply(data1, target)

        data1 = CorrelationMatrix().apply(data1, target)

        if self.with_eigen:
            w = Eigenvalues().apply(data1, target)

        out = []
        if self.with_corr:
            data1 = upper_right_triangle(data1)
            out.append(data1)
        if self.with_eigen:
            out.append(w)

        for d in out:
            assert d.ndim == 1

        return np.concatenate(out, axis=0)


class TimeFreqCorrelation:
    """
    Combines time and frequency correlation, taking both correlation coefficients and eigenvalues.
    """
    def __init__(self, start, end, max_hz, scale_option):
        self.start = start
        self.end = end
        self.max_hz = max_hz
        self.scale_option = scale_option
        assert scale_option in ('us', 'usf', 'none')

    def get_name(self):
        return 'time-freq-correlation-%d-%d-r%d-%s' % (self.start, self.end, self.max_hz, self.scale_option)

    def apply(self, data, target):
        data1 = TimeCorrelation(self.max_hz, self.scale_option).apply(data, target)
        data2 = FreqCorrelation(self.start, self.end, self.scale_option).apply(data, target)
        assert data1.ndim == data2.ndim
        return np.concatenate((data1, data2), axis=data1.ndim-1)

class RawData:
    """
    Combines time and frequency correlation, taking both correlation coefficients and eigenvalues.
    """
    def __init__(self):
        return

    def get_name(self):
        return 'RawData'

    def apply(self, data, target):
        return data
    
    
class FFTWithTimeFreqCorrelation:
    """
    Combines FFT with time and frequency correlation, taking both correlation coefficients and eigenvalues.
    """
    def __init__(self, start, end, max_hz, scale_option):
        self.start = start
        self.end = end
        self.max_hz = max_hz
        self.scale_option = scale_option

    def get_name(self):
        return 'fft-with-time-freq-corr-%d-%d-r%d-%s' % (self.start, self.end, self.max_hz, self.scale_option)

    def apply(self, data, target):
        data1 = TimeCorrelation(self.max_hz, self.scale_option).apply(data, target)
        data2 = FreqCorrelation(self.start, self.end, self.scale_option, with_fft=True).apply(data, target)
        assert data1.ndim == data2.ndim
        return tf.concat([data1, data2], axis=data1.ndim-1)

class GSPFeatures:
    
    def __init__(self,choose_or_not):
        self.choose_or_not = choose_or_not
         
        
        
    def get_name(self):
        return 'GSP'
    
    def GSP_feature_select(self,choose_or_not):
        topology_feature_num = 36
        GSP_feature_each = 10
        GSP_wavelet_each = GSP_feature_each + 3
        howmany_max = 2 # <= GSP_feature_each
        howmany_min = 2 # <= GSP_feature_each
        howmany_wavelet = 5 # <= GSP_wavelet_each
        if(choose_or_not == 'v_1'):
            features = np.array([41,53,38,39,40,50]) #patient1
        if(choose_or_not == 'v_2'):
            features = np.array([11,255,274,218,213,200,182,181,180,140,120,119,76])
        if(choose_or_not == 'v_3'):
            features = np.array([5,6,11,35,40,41,47,58,59,69,108,109,115,116,117,118,119,121,129,133,135,152,238,244,264])
            features1 = np.array([6,18,30,36,56,57,95,135,147,173,274,277,316]) # patient_1 : ****147, **18,6  *56,57,30,36
            features5 = np.array([51,6,11,12,18,19,24,30,31,35,36,37,38,40,45,47,48,69,70,81,86,95,108,316]) # patient_5
            features3 = np.array([6,47,69,144])
            features = np.array([47,11,6,18,30,36,51,57,95,135,147]) #40 is bad for p1, 108,69 is bad for all, 
        if(choose_or_not == 'v_5'):
            features = np.array([0,1,2,3,4,7,8,9,15,16,17]) # 0,1,2,3,4,7,8,9,15,16,17
#             features = np.unique(np.hstack((np.intersect1d(features, features1),np.intersect1d(features1, features5))) )
#             features = features5
        if(choose_or_not == 'v_6'):
            features = np.array([3,4,8,9,39,   20,25,27,28,33,36,38, 43]) #20,25,27,28,33,36,38, 43
            features = np.unique(np.hstack((features , np.arange(49))) )
            features = np.setdiff1d( np.arange(49),np.array([10,13,22,34]))
        if(choose_or_not == 'v_7'):    
            features = []
            each_band_num_feature = 49
            for freq_band in [1,2]: #0,1,2,3,4
                features = np.hstack((features, np.hstack((np.arange(12),np.arange(13+12)+24))+freq_band*each_band_num_feature ))
#                 features = np.hstack((features, np.arange(each_band_num_feature)+freq_band*each_band_num_feature ))
        if(choose_or_not == 'v_77'):    
            features = np.setdiff1d( np.arange(125)+125,np.hstack((np.array([10,13,22,34]),np.array([10,13,22,34])+49,\
                                                                                               np.array([10,13,22,34])+49*2,\
                                                                                                         np.array([10,13,22,34])+49*3,\
                                                                                                         np.array([10,13,22,34])+49*4)))
            features = np.hstack(( np.arange(125)+125))
        if(choose_or_not == 'v_777'):
            features = []
            for freq_band in [0,1,2]: #0,1,2,3,4
                features = np.hstack((features , np.setdiff1d(np.hstack(( np.arange(10)+freq_band*188,np.array([12,22,26,40,44])+freq_band*188-1\
                                                                          ,np.arange(188-64)+freq_band*188+64)), np.array([180])+freq_band*188-1)))
        if(choose_or_not == 'v_9'):
            features = []
            each_band_num_feature = 656
            for freq_band in [1,2]: #0,1,2,3,4
#                 features = np.hstack((features , np.setdiff1d(np.hstack(( np.arange(10)+freq_band*each_band_num_feature,np.array([12,22,26,40,44])+freq_band*each_band_num_feature-1\
#                                                                           ,np.arange(each_band_num_feature-64)+freq_band*each_band_num_feature+64)), np.array([180])+freq_band*each_band_num_feature-1)))
                features = np.hstack((features, np.arange(each_band_num_feature)+freq_band*each_band_num_feature ))
        if(choose_or_not == 'v_10'):
            features = []
            each_band_num_feature = 96
            for freq_band in [1,2]: #0,1,2,3,4
                features = np.hstack((features, np.arange(each_band_num_feature)+freq_band*each_band_num_feature ))
        
        if(choose_or_not == 'Dog_1'):
            features = []
            each_band_num_feature = 145
            for freq_band in [2,3,4,5,6,7]: #0,1,2,3,4
                features = np.hstack((features, np.arange(12)+freq_band*each_band_num_feature, np.arange(12)+freq_band*each_band_num_feature+12, np.arange(12)+freq_band*each_band_num_feature+24, np.arange(4)+freq_band*each_band_num_feature+36  , - np.arange(20)+(freq_band+1)*each_band_num_feature  ))
            features = np.hstack((features , 1160 + np.arange(0 )))       
        
        if(choose_or_not == 'Dog_2'):
            features = []
            each_band_num_feature = 145
            for freq_band in [0,1,2,3,4,5,6,7]: #0,1,2,3,4
                features = np.hstack((features, np.arange(12)+freq_band*each_band_num_feature, np.arange(12)+freq_band*each_band_num_feature+12, np.arange(6)+freq_band*each_band_num_feature+24, np.arange(0)+freq_band*each_band_num_feature+36  , - np.arange(20)+(freq_band+1)*each_band_num_feature  ))
            features = np.hstack((features , 1160 + np.arange(12)))       
        
        if(choose_or_not == 'Dog_3'):
            features = []
            each_band_num_feature = 145
            eigenval = 4;
            graphLap = 2;
            wavelet = 4;
            energy_bands = 99;
            for freq_band in [0,1,2,3,4,5,6,7]: #0,1,2,3,4
                features = np.hstack((features, np.arange(12)+freq_band*each_band_num_feature, np.arange(12)+freq_band*each_band_num_feature+12, np.arange(12)+freq_band*each_band_num_feature+24, \
                                      np.arange(4)+freq_band*each_band_num_feature+36 , np.arange(0)+freq_band*each_band_num_feature+36+eigenval , \
                                      np.arange(5)+(freq_band)*each_band_num_feature+36+eigenval+graphLap ,  - np.arange(5)+(freq_band)*each_band_num_feature+36+eigenval+graphLap+wavelet ,\
                                      np.arange(40)+(freq_band)*each_band_num_feature+36+eigenval+graphLap+wavelet , -np.arange(40)+(freq_band)*each_band_num_feature+36+eigenval+graphLap+wavelet+energy_bands ))
            features = np.hstack((features , 1160 + np.arange(12 ))) #1881
        
        
#         if(choose_or_not == 'Dog_3'):
#             features = []
#             each_band_num_feature = 209
#             eigenval = 4
#             graphLap = 10
#             wavelet = 60
#             energy_bands = 99
#             for freq_band in [0,1,2,3,4,5,6,7]: #0,1,2,3,4
#                 features = np.hstack((features, np.arange(12)+freq_band*each_band_num_feature, np.arange(12)+freq_band*each_band_num_feature+12, np.arange(12)+freq_band*each_band_num_feature+24, \
#                                       np.arange(4)+freq_band*each_band_num_feature+36 , np.arange(10)+freq_band*each_band_num_feature+36+eigenval , \
#                                       np.arange(5)+(freq_band)*each_band_num_feature+36+eigenval+graphLap ,  - np.arange(5)+(freq_band)*each_band_num_feature+36+eigenval+graphLap+wavelet ,\
#                                       np.arange(40)+(freq_band)*each_band_num_feature+36+eigenval+graphLap+wavelet , -np.arange(40)+(freq_band)*each_band_num_feature+36+eigenval+graphLap+wavelet+energy_bands ))
#             features = np.hstack((features , 1672 + np.arange(12 ))) 
        
        
            
        if(choose_or_not == 'Dog_4'):
            features = []
            each_band_num_feature = 145 #209
            eigenval = 4
            graphLap = 10
            wavelet = 60
            energy_bands = 99
            for freq_band in [0,1,2,3,4,5,6,7]: #0,1,2,3,4
                features = np.hstack((features, np.arange(12)+freq_band*each_band_num_feature, np.arange(0)+freq_band*each_band_num_feature+12, np.arange(0)+freq_band*each_band_num_feature+24, \
                                      np.arange(0)+freq_band*each_band_num_feature+36 , np.arange(0)+freq_band*each_band_num_feature+36+eigenval , \
                                      np.arange(0)+(freq_band)*each_band_num_feature+36+eigenval+graphLap ,  - np.arange(0)+(freq_band)*each_band_num_feature+36+eigenval+graphLap+wavelet ,\
                                      np.arange(0)+(freq_band)*each_band_num_feature+36+eigenval+graphLap+wavelet , -np.arange(40)+(freq_band)*each_band_num_feature+36+eigenval+graphLap+wavelet+energy_bands ))
            features = np.hstack((features , 1160 + np.arange(0 )))     
        
        
        if(choose_or_not == 'Patient_1'):
            features = []
            each_band_num_feature = 145
            for freq_band in [0,1,2,3,4]: # a little different from adding the rest
                features = np.hstack((features, np.arange(12)+freq_band*each_band_num_feature, np.arange(12)+freq_band*each_band_num_feature+12, np.arange(12)+freq_band*each_band_num_feature+24, np.arange(4)+freq_band*each_band_num_feature+36 , np.arange(50)+(freq_band+3/5)*each_band_num_feature , - np.arange(20)+(freq_band+1)*each_band_num_feature  ))
            features = np.hstack((features , 1160 + np.arange(12 )))       
        
        
         
        if(choose_or_not == 'Patient_2'):
            features = []
            each_band_num_feature = 145
            for freq_band in [0,1,2,3,4,5,6,7]: #0,1,2,3,4
                features = np.hstack((features, np.arange(12)+freq_band*each_band_num_feature, np.arange(12)+freq_band*each_band_num_feature+12, np.arange(12)+freq_band*each_band_num_feature+24, np.arange(4)+freq_band*each_band_num_feature+36 , np.arange(0)+(freq_band+3/5)*each_band_num_feature , - np.arange(30)+(freq_band+1)*each_band_num_feature  ))
            features = np.hstack((features , 1160 + np.arange(12 )))       
            
            
        if(choose_or_not == 'Patient_3'):
            features = []
            each_band_num_feature = 145
            for freq_band in [4,5,6,7]: #0,1,2,3,4
                features = np.hstack((features, np.arange(12)+freq_band*each_band_num_feature, np.arange(12)+freq_band*each_band_num_feature+12, np.arange(12)+freq_band*each_band_num_feature+24, np.arange(4)+freq_band*each_band_num_feature+36 , - np.arange(20)+(freq_band+1)*each_band_num_feature  ))
            features = np.hstack((features , 1160 + np.arange(12 )))       
        
#         if(choose_or_not == 'Patient_3'):
#             features = []
#             each_band_num_feature = 209 #209
#             eigenval = 4
#             graphLap = 10
#             wavelet = 60
#             energy_bands = 99
#             for freq_band in [4,5,6,7]: #0,1,2,3,4
#                 features = np.hstack((features, np.arange(12)+freq_band*each_band_num_feature, np.arange(3)+freq_band*each_band_num_feature+12, np.arange(0)+freq_band*each_band_num_feature+24, \
#                                       np.arange(0)+freq_band*each_band_num_feature+36 , np.arange(0)+freq_band*each_band_num_feature+36+eigenval , \
#                                       np.arange(0)+(freq_band)*each_band_num_feature+36+eigenval+graphLap ,  - np.arange(0)+(freq_band)*each_band_num_feature+36+eigenval+graphLap+wavelet ,\
#                                       np.arange(0)+(freq_band)*each_band_num_feature+36+eigenval+graphLap+wavelet , -np.arange(40)+(freq_band)*each_band_num_feature+36+eigenval+graphLap+wavelet+energy_bands ))
#             features = np.hstack((features , 1160 + np.arange(12)))  
         
         
        if(choose_or_not == 'Patient_4'):
            features = []
            each_band_num_feature = 145
            for freq_band in [3,4,5,6,7]: #0,1,2,3,4
                features = np.hstack((features, np.arange(12)+freq_band*each_band_num_feature, np.arange(3)+freq_band*each_band_num_feature+12, np.arange(0)+freq_band*each_band_num_feature+24, np.arange(0)+freq_band*each_band_num_feature+36  , - np.arange(0)+(freq_band+1)*each_band_num_feature  ))
            features = np.hstack((features , 1160 + np.arange(12 )))       
        
        
           
            
        if(choose_or_not == 'Patient_5'):
            features = []
            each_band_num_feature = 145
            for freq_band in [3,4,5,6,7]: #0,1,2,3,4
                features = np.hstack((features, np.arange(12)+freq_band*each_band_num_feature, np.arange(0)+freq_band*each_band_num_feature+12, np.arange(12)+freq_band*each_band_num_feature+24, np.arange(0)+freq_band*each_band_num_feature+36 , - np.arange(0)+(freq_band+1)*each_band_num_feature  ))
            features = np.hstack((features , 1160 + np.arange(12 )))       
        
        if(choose_or_not == 'Patient_6'):
            features = []
            each_band_num_feature = 209
            eigenval = 4;
            graphLap = 10;
            wavelet = 60;
            energy_bands = 99;
            for freq_band in [0,1,2,3,4,5,6,7,8]: #0,1,2,3,4
                features = np.hstack((features, np.arange(12)+freq_band*each_band_num_feature, np.arange(12)+freq_band*each_band_num_feature+12, np.arange(12)+freq_band*each_band_num_feature+24, \
                                      np.arange(4)+freq_band*each_band_num_feature+36 , np.arange(0)+freq_band*each_band_num_feature+36+eigenval , \
                                      np.arange(10)+(freq_band)*each_band_num_feature+36+eigenval+graphLap ,  - np.arange(10)+(freq_band)*each_band_num_feature+36+eigenval+graphLap+wavelet ,\
                                      np.arange(15)+(freq_band)*each_band_num_feature+36+eigenval+graphLap+wavelet , - np.arange(15)+(freq_band)*each_band_num_feature+36+eigenval+graphLap+wavelet+energy_bands ))
            features = np.hstack((features , 1881 + np.arange(12 ))) 
        
        
        
#         if(choose_or_not == 'Patient_6'):
#             features = []
#             each_band_num_feature = 145
#             for freq_band in [0,1,2,3,4,5,6,7]: # a little different from adding the rest
#                 features = np.hstack((features, np.arange(12)+freq_band*each_band_num_feature, np.arange(0)+freq_band*each_band_num_feature+12, np.arange(3)+freq_band*each_band_num_feature+24, np.arange(0)+freq_band*each_band_num_feature+36 , np.arange(0)+(freq_band)*each_band_num_feature + 36 + 60 , - np.arange(0)+(freq_band+1)*each_band_num_feature  ))
#             features = np.hstack((features , 1160 + np.arange(12)))       
        
        
        if(choose_or_not == 'Patient_7'):
            features = []
            each_band_num_feature = 145
            for freq_band in [3,4,5,6,7]: #0,1,2,3,4
                features = np.hstack((features, np.arange(12)+freq_band*each_band_num_feature, np.arange(12)+freq_band*each_band_num_feature+12, np.arange(12)+freq_band*each_band_num_feature+24, np.arange(14)+freq_band*each_band_num_feature+36  , - np.arange(20)+(freq_band+1)*each_band_num_feature  ))
            features = np.hstack((features , 1160 + np.arange(12 )))       
        
        if(choose_or_not == 'Patient_8'):
            features = []
            each_band_num_feature = 145
            for freq_band in [3,4,5,6,7]: #0,1,2,3,4
                features = np.hstack((features, np.arange(12)+freq_band*each_band_num_feature, np.arange(12)+freq_band*each_band_num_feature+12, np.arange(12)+freq_band*each_band_num_feature+24, np.arange(14)+freq_band*each_band_num_feature+36  , - np.arange(20)+(freq_band+1)*each_band_num_feature  ))
            features = np.hstack((features , 1160 + np.arange(12 )))       
        
        
        
        
#         features = np.hstack((features , topology_feature_num + np.arange(howmany_max )))
#         features = np.hstack((features , topology_feature_num + GSP_feature_each + np.arange(howmany_min )))
#         features = np.hstack((features , topology_feature_num + GSP_feature_each * 2 + np.arange(howmany_wavelet * howmany_max )))
#         features = np.hstack((features , topology_feature_num + GSP_feature_each * 2 + GSP_wavelet_each * GSP_feature_each + np.arange(howmany_wavelet * howmany_min )))
#         features = np.hstack((features , GSP_wavelet_each * GSP_feature_each *2 +GSP_feature_each*2))
        return features.astype(int)
    
    def re_window (self, data):
        WindowSize = 100
        data_out=-1
        each_band_num_feature = 145
        for freq_band in [0,1,2,3,4,5,6,7]:
            for w in range(int(100/WindowSize)):
                data_out = np.hstack((data_out , np.mean(data[:,np.arange(WindowSize)+freq_band*each_band_num_feature+45 +w*WindowSize]) ))
        return data_out[1:]
    
    def apply(self, data, target):
        if(self.choose_or_not != 'all'):
            if(0):
                data = np.hstack ((data[:,self.GSP_feature_select(target)], self.re_window (data)[np.newaxis,:]))
            else:
                data = data[:,self.GSP_feature_select(target)]
        return data


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a    

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def energy(data, lowcut, highcut, fs, order = 5):
    oneD_array_of_amps = butter_bandpass_filter(data, lowcut, highcut, fs, order=order)
    #calculate energy like this
    energy = sum([x*2 for x in oneD_array_of_amps])
    return energy 
 
class Energy:
    def __init__(self, lowcut, highcut, fs, order=5, axis=-1):
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order
        self.axis = axis
        

    def get_name(self):
        return "Energy"

    def apply(self, data, target=None):
        
        return np.sum(data, axis=self.axis_to_sum)

