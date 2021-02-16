import json
import os.path
import numpy as np
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.fftpack import rfft, irfft, fftfreq
from graphsage.utils import normal_SDP_solver, Quadratic_SDP_solver
from graphsage.unsupervised_train import main
import pyedflib
import networkx as nx
#from pylab import *
from sklearn.cluster import DBSCAN
from sklearn.decomposition import NMF
from sklearn import metrics
#from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0, '..\Detection_Project')
# from seizure.tasks import TaskCore, train_classifier
from utils import softmax, convert, hybrid_repres_estimation, json_gen, Graph_Construction,\
                                 EigenVec_Centrality, plotting,plot_edges,plot_nodes, coherence_func, plot_eeg, initial_feature_vector_brain
from PIL import Image

# a=np.array([96,24,52,24,53,52,18,28,24,52,18,15,15,52,24,66,24,24,15,53,24,42,24,13\
# ,24,66,24,13,59,13,24,24,63,24,24,24,24,24,74,28,91,24,52,24,15,52,24,66\
# ,24,12,53,31,24,24,24,63,5,36,74,52,42,24,15,13,24,53,22,42,41,36,42,41\
# ,41,41,42,42,5,5,12,56,56,53,24,74,11,17,24,24,24,46,66,41,42,42,66,41\
# ,42,15,36,42])
# set_a = np.array(list(set(a)))
# for ii in range(len(set_a)):
#     a = np.where(a==set_a[ii],ii+1,a)

num_epochs = 30
inner_sampled_length = None # 10

initial_freq_band = 4
freq_bands = [9,25]#[4,8,12,18,24,30,100]#
iteration_flag = 1 # 3
theta_exist = True
flag_normalized = False    

add_phase = 0
add_amp = 1
howmany_to_load = 12
l_num = 1e10
targets = [ #'Dog_1', 'Patient_3'
    'chb01',
]

input_prefix =  'C:/1My files/MIT data set' #'/home/ghnaf/MIT data set' #  
ouput_prefix = "output_MITDB/"

aggregator_models = ['graphsage_mean', 'graphsage_maxpool', 'gcn', 'graphsage_seq', 'graphsage_meanpool', 'n2v']
aggregator_model = 'graphsage_mean'
state_estimation_modes = ['clustering_DBSCAN', 'NMF']
state_estimation_mode = 'NMF'
#0:invcov, 1:corr, 2:coherence, 3:custom, 4:custom_theta, 5:hybrid_repres
name_weight_matrices = ['inv_cov','correlation','coherence','custom', 'custom_theta', 'hybrid_representation_'+aggregator_model]
num_approaches = len(name_weight_matrices)
graph_const_modes = ['full', 'random']
graph_const_mode = 'random'
population_coeff = 1
initial_feature_mode = 'Cor+PLV+Coh'
num_windows = 2
# num_windows += 1


def network_density(a, th):
    N = a.shape[0]
#     a = np.where(a>th, 1, 0)
    return (np.sum(a) - np.trace(a))/(2*N*(N-1))
    
    

def post_processing_target(save_filename, interictal_features, ictal_features):
    plt.figure()
    
    plt.savefig(save_filename + '_activity' + '.png')
    return
    plt.figure()
    for i in np.arange(int(interictal_features[0].shape[0]/2))*2:
        plt.scatter([np.linalg.eig(np.where(a<l_num,a,l_num))[1][:,0][i] for a in interictal_features], \
                    [np.linalg.eig(np.where(a<l_num,a,l_num))[1][:,0][i+1] for a in interictal_features], s=80, marker="o") 
        plt.scatter([np.linalg.eig(np.where(a<l_num,a,l_num))[1][:,0][i] for a in ictal_features], \
                    [np.linalg.eig(np.where(a<l_num,a,l_num))[1][:,0][i+1] for a in ictal_features], s=80, marker="+")
    plt.savefig(save_filename + '_activity' + '.png')
    return
    interictal_mean_vals = [np.mean(a) for a in interictal_features]
    interictal_min_vals = [np.min(a) for a in interictal_features]
    interictal_max_vals = [np.max(a) for a in interictal_features]
    ictal_mean_vals = [np.mean(a) for a in ictal_features]
    ictal_min_vals = [np.min(a) for a in ictal_features]
    ictal_max_vals = [np.max(a) for a in ictal_features]
    plt.figure()
    plt.scatter(interictal_min_vals, interictal_max_vals, s=80, marker="o") 
    plt.scatter(ictal_min_vals, ictal_max_vals, s=80, marker="+")
    plt.savefig(save_filename + '_minmax' + '.png')
    plt.figure()
    plt.scatter(interictal_mean_vals, interictal_max_vals, s=80, marker="o") 
    plt.scatter(ictal_mean_vals, ictal_max_vals, s=80, marker="+")
    plt.savefig(save_filename + '_meanmax' + '.png')
    plt.figure()
    plt.scatter(interictal_mean_vals, interictal_min_vals, s=80, marker="o") 
    plt.scatter(ictal_mean_vals, ictal_min_vals, s=80, marker="+")
    plt.savefig(save_filename + '_meanmin' + '.png')
    return

def change_file_name(name):
    out_name = '_'.join((name.split('_')[:4])) + '_' + str(int(name.split('_')[-1]))
    return out_name


def inner_loadedf(input_prefix):
    f = pyedflib.EdfReader(input_prefix)
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    sigbufs = np.zeros((n, f.getNSamples()[0]))
    for i in np.arange(n):
        sigbufs[i, :] = f.readSignal(i)
    return sigbufs

def compare(a,b):
    out = np.zeros(a.shape)
    for i in len(a):
        out[i]
def seizure_loadedf(comp_file_name):
    text_file = open(comp_file_name + '.seizures', "rb")
    byte_array = text_file.read()
    byte_array_chars = np.array(list(byte_array))
#     bytearray(byte_array)
#     byte_array_chars = list(byte_array)    
#     compare(byte_array , int('ec', 16))
#     words = zip(byte_array , int('ec', 16),int('ec', 16))
#     np.where(byte_array == int('ec', 16), 1, 0)
#     incorrect=len([c for c,d in words if c!=d])
    
    number_of_seizures = int((np.sum(byte_array_chars == int('ec', 16) ) - 1) / 2)
    seizure_start_time_offsets = np.zeros((number_of_seizures,))
    seizure_lengths = np.zeros((number_of_seizures,))
    for i in np.arange(number_of_seizures)+1:
        seizure_start_time_offsets[i-1] = int( bin(byte_array_chars[22 + (i*16)])[2:] + bin(byte_array_chars[25 + (i*16)])[2:] , 2)
        seizure_lengths[i-1] = byte_array_chars[33 + (i*16)]
    text_file.close()
    return seizure_start_time_offsets, seizure_lengths



    
def load_edf_data(input_prefix, target):
    dir = input_prefix + '/' + target
#     comp_file_name = dir + '/' + 'chb01_03'+ '.edf'
#     data = inner_loadedf( comp_file_name )
#     if os.path.exists(comp_file_name + '.seizures'):
#         seizure_start_time_offsets, seizure_lengths = seizure_loadedf( comp_file_name )
#     else:
#         seizure_start_time_offsets = -1
#         seizure_lengths = -1
#     print('settings: %f, %f' % (seizure_start_time_offsets, seizure_lengths ))
#     yield(data, comp_file_name, seizure_start_time_offsets, seizure_lengths)
#     return    
    filenames = sorted(os.listdir(dir))
    counter_to_load = 0
    for i, filename in enumerate(filenames):
        comp_file_name = dir + '/' + filename
        if(filename.find('.edf') != -1 and filename.find('.seizures') == -1):
            data = inner_loadedf( comp_file_name)
                        
            if os.path.exists(comp_file_name + '.seizures'):
                seizure_start_time_offsets, seizure_lengths = seizure_loadedf( comp_file_name )
            else:
                continue
                seizure_start_time_offsets = -1
                seizure_lengths = -1
            print('settings: %f, %f' % (seizure_start_time_offsets, seizure_lengths ))
            yield(data, filename.split('.')[0], seizure_start_time_offsets, seizure_lengths)
            
            
            if(howmany_to_load != -1):
                if(counter_to_load == howmany_to_load):
                    break
                else:
                    counter_to_load += 1
    
    
              

def load_mat_data(input_prefix, target, data_type):
    dir = os.path.join(input_prefix, target)
    
    filenames = sorted(os.listdir(dir))
    counter_to_load = 0
    for i, filename in enumerate(filenames):
        comp_file_name = dir + '/' + filename
        #print(comp_file_name)
        if(filename.find('_' + data_type) != -1):
            data = loadmat( comp_file_name)
            new_file_name = change_file_name(filename.split('.')[0])
            yield(data, new_file_name)
            if(howmany_to_load != -1):
                if(counter_to_load == howmany_to_load):
                    break
                else:
                    counter_to_load += 1
            
def down_sampling(X, ratio):
    S,T = X.shape
    output = np.zeros((S,ratio))
    window_len = int(T/ratio)
    for i in range(ratio):
        output[:,i] = np.mean(X[:,i * window_len:(i+1) * window_len], axis=1)    
    return output


# def calc_correlation(data1, data2):
#     centered_data = data1 - np.repeat(np.mean(data1, axis=1)[:,np.newaxis], [T], axis=1)
#     np.divide(np.matmul(centered_data, centered_data.T), np.repeat(np.sum(np.power(centered_data,2), axis=1)[:,np.newaxis], [S], axis=1))
  


def convert_weight_features(a):
    N,_ = a.shape
    return np.reshape(a[np.triu_indices(N)],(int(N*(N+1)/2),)) 
      
def   W_generator(data_all, file_name, seizure_start_time_offsets, seizure_lengths, target, out_dir, out_edf_data, counter_data, flag_normalized, Theta_in=None, U_in=None):
    S,T = data_all.shape
    num_nodes = S 
    sampling_freq = 256
    if(num_windows is not None):
        win_len_sec = T/(sampling_freq * num_windows)
    else:
        win_len_sec = 2.5
    stride_sec = 1.5 # win_len_sec  
    print('win_len_sec: ', win_len_sec)
    
    
    A = Graph_Construction(num_nodes, graph_const_mode, population_coeff)
#     A = (A-np.diag(np.diag(A)))

    
    seizure_start_time_offsets *= sampling_freq
    seizure_lengths *= sampling_freq
    
    win_len = np.ceil( win_len_sec * sampling_freq)
    stride = np.ceil( stride_sec * sampling_freq)
    intervals_with_stride = np.arange(0, T - win_len, stride)
    
    intervals_seizures = np.zeros(intervals_with_stride.shape)
    for win_ind in range(len(intervals_with_stride)):
        w = intervals_with_stride[win_ind]
        time_inter = (w + np.arange(win_len)).astype(np.int)
        if(seizure_start_time_offsets>=0 and seizure_lengths>=0):
            if( not( (seizure_start_time_offsets>=w + win_len) or ( seizure_start_time_offsets + seizure_lengths <=w)  )):
                intervals_seizures[win_ind] = 1
    
    
    weight_matrices = []
    for _ in range(num_approaches):
        weight_matrices.append([])
    weight_matrices_features = np.zeros((num_approaches, len(intervals_with_stride), int(num_nodes*(num_nodes+1)/2)))
    homany_before_show = 10
    for win_ind in range(len(intervals_with_stride)):
        if(not (intervals_seizures[win_ind]==1 
            or (win_ind+homany_before_show<=len(intervals_with_stride) and np.any(intervals_seizures[win_ind+1:win_ind+homany_before_show]==1)))):
#                     or (win_ind-5>=0 and np.any(intervals_seizures[win_ind-5:win_ind]==1)))):
           
            continue
        w = intervals_with_stride[win_ind]
        time_inter = (w + np.arange(win_len)).astype(np.int)
#         if(seizure_start_time_offsets>=0 and seizure_lengths>=0):
#             if( not( (seizure_start_time_offsets>=w + win_len) or ( seizure_start_time_offsets + seizure_lengths <=w)  )):
#                 intervals_seizures[win_ind] = 1
        
        data = data_all[:,time_inter]
        
        T = data.shape[1]
        S = data.shape[0]
        try:
            inv_cov = np.linalg.inv(np.matmul(data, data.T))
        except:
            inv_cov = np.linalg.pinv(np.matmul(data, data.T))
            
        corr_mat = np.corrcoef(data)
        coherence_mat =  coherence_func(data, [4,25], sampling_freq, sampling_freq, 0.75) 
        
        if(inner_sampled_length is not None):
            data = down_sampling(data, int(inner_sampled_length))
        #print('Down-Sampled size: ', data.shape[1])
        data_ext, sizes = initial_feature_vector_brain (initial_feature_mode, data, sampling_freq, freq_bands, initial_freq_band)       
        hybrid_repres_W, Theta_out, _, _, _, U_out = hybrid_repres_estimation(data_ext, A, input_prefix, ouput_prefix, file_name, \
                                                                                        aggregator_model, num_epochs, iteration_flag, \
                                                                                        theta_exist, flag_normalized, brain_similarity_sizes=sizes,\
                                                                                        loss_function='brain_hybrid_loss')
#         hybrid_repres_W = hybrid_repres_W
        _,D = data_ext.shape
        Z_tilde = np.repeat(data_ext, [num_nodes], axis=0)
        Z_tilde_tilde = np.tile(data_ext, (num_nodes,1)) 
        final_theta_1 = Quadratic_SDP_solver (Z_tilde, Z_tilde_tilde, D, D) 
        #final_theta_1 = np.eye(D)
        W_custom = np.corrcoef(data_ext)
        W_custom_theta = np.matmul(data_ext, np.matmul(final_theta_1, data_ext.T))
        
#         weight_matrices_features[0,win_ind,:] = convert_weight_features(inv_cov)     
#         weight_matrices_features[1,win_ind,:] = convert_weight_features(corr_mat) #
#         weight_matrices_features[2,win_ind,:] = convert_weight_features(coherence_mat)
#         weight_matrices_features[3,win_ind,:] = convert_weight_features(W_custom)
#         weight_matrices_features[4,win_ind,:] = convert_weight_features(W_custom_theta)
#         weight_matrices_features[5,win_ind,:] = convert_weight_features(hybrid_repres_W)
        
        weight_matrices[0].append(inv_cov)    
        weight_matrices[1].append(corr_mat)
        weight_matrices[2].append(coherence_mat)
        weight_matrices[3].append(W_custom)
        weight_matrices[4].append(W_custom_theta)
        weight_matrices[5].append(hybrid_repres_W)
        
        print('window number: ', len(weight_matrices[0])) 
        
    print(file_name +' is done!')
    counter_data += 1
    return counter_data, weight_matrices_features, weight_matrices,intervals_seizures , Theta_out, U_out


def state_estimation(weight_matrices, file_name, state_estimation_mode):
    X = weight_matrices.T
#     X = StandardScaler().fit_transform(X)
    if(state_estimation_mode == 'clustering_DBSCAN'):
        model = DBSCAN().fit(X) #eps=0.3, min_samples=10
        hard_states = model.labels_
        n_clusters = len(set(hard_states)) - (1 if -1 in hard_states else 0)
        
    elif(state_estimation_mode == 'NMF'):
        model = NMF(n_components=min(10,num_windows-1)) #alpha = 0.1, l1_ratio =0.1 #n_components=min(10,num_windows)
        try:
            W = model.fit_transform(X-np.min(X))
            H = model.components_ #array, [n_components, n_features]
            #         print(H.shape)
            #         print(W.shape)
#             print(np.argmax(H,axis=0))
#             print(np.argmax(H,axis=1))
#             print(np.argmax(W,axis=0))
#             print(np.argmax(W,axis=1))
            hard_states = np.argmax(H,axis=0)
        except Exception as e:
            hard_states = np.zeros((weight_matrices.shape[0],))
    n_states = len(set(hard_states))
    print(file_name)
    print('    Estimated number of states: %d' % n_states)
    print('    state trajectory: ', hard_states)
    return hard_states


def map_states(a):   
    set_a = np.array(list(set(a)))
    for ii in range(len(set_a)):
        a = np.where(a==set_a[ii],ii+1,a)
    return a

if __name__ == '__main__':
    for target in targets:             
        out_dir = os.path.join(ouput_prefix, target)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
        out_edf_data = load_edf_data(input_prefix, target) 
        counter_data = 1
        Theta = None
        U = None
        for data_all, file_name, seizure_start_time_offsets, seizure_lengths in out_edf_data:
#             plot_eeg(data_all, seizure_start_time_offsets, seizure_lengths)
            counter_data, weight_matrices_features, weight_matrices,intervals_seizures, Theta, U  = \
                        W_generator(data_all, file_name, seizure_start_time_offsets, seizure_lengths, target, out_dir, out_edf_data, counter_data, flag_normalized, Theta_in=Theta, U_in=U)
            estimated_states = None
            for appr in range(num_approaches):
#                 estimated_states = state_estimation(weight_matrices_features[appr,:,:], file_name, state_estimation_mode)
#                 estimated_states = map_states(estimated_states)
                plotting(out_dir + '/' + name_weight_matrices[appr], file_name, weight_matrices[appr], intervals_seizures = intervals_seizures, estimated_states = estimated_states)
                 



