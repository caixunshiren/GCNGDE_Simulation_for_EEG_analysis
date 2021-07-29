import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power
from DataManager import *


#shows heatmap a 2D matrix
def show_heat_map(m, cmap = 'bwr'):
    plt.imshow(m, cmap=cmap, interpolation='nearest')
    plt.colorbar()
    plt.show()

def create_DAD(adj):
    np.fill_diagonal(adj, 1)
    rowsum = np.sum(adj, axis=1)
    d = np.diag_indices(adj.shape[0])
    D = np.zeros(adj.shape)
    D[d] = rowsum
    #print(rowsum)
    D = fractional_matrix_power(D, -0.5)
    #print(D)
    return D@adj@D

def shuffle_train_test(variables, train_ratio = 0.7, indices = None, print_summary = True):
    data_all = np.concatenate((variables["X_train"], variables["X_test"]), axis = 2)
    labels_all = np.concatenate((variables["y_train"], variables["y_test"]), axis = 1)
    if indices is None:
        indices = np.random.permutation(data_all.shape[2])
    split_ind = int(train_ratio * data_all.shape[2])
    training_idx, test_idx = indices[:split_ind], indices[split_ind:]
    #print(split_ind, training_idx.shape)
    variables["X_train"], variables["X_test"] = data_all[:,:,training_idx], data_all[:,:,test_idx]
    variables["y_train"], variables["y_test"] = labels_all[:,training_idx], labels_all[:,test_idx]
    if print_summary:
        print("X_train:", variables["X_train"].shape)
        print("X_test:", variables["X_test"].shape)
        print("y_train:", variables["y_train"].shape, "Positive labes:", np.sum(variables["y_train"]))
        print("y_test:", variables["y_test"].shape, "Positive labes:",np.sum(variables["y_test"]))
    return variables, indices

def remove_overlap(variables):
    variables["X_train"], variables["X_test"] = variables["X_train"][192:449,:,:], variables["X_test"][192:449,:,:]
    return variables

def load_patient_data(filepath, verbose = True):
    variables = {}
    f = h5py.File(filepath, 'r')
    for k, v in f.items():
        variables[k] = np.array(v)
        '''
        if k == 'X_train' or k == 'X_test':
            if len(variables[k].shape) == 4:
                shape = variables[k].shape
                variables[k] = np.reshape(variables[k], (shape[0]*shape[1], shape[2], shape[3]))
        '''
        
        
    if verbose: 
        for key in variables.keys():
            print("name:", key)
            print("shape:", variables[key].shape)
            print("-----------------------")
        
        print('print sample EGG signal from one node:')
        '''
        plt.figure()
        ax = plt.axes()
        X_train = variables["X_train"]
        X_test = variables["X_test"]
        ax.plot(np.linspace(0, 10, X_train.shape[0]), X_train[:,5,0])
        '''
        plot_signal(variables["X_train"], 5,0)
        #plot_signal(variables["X_train"], 5,1)
        #plot_signal(variables["X_train"], 5,2)
    return variables

def plot_signal(data, node, sample):
    plt.figure()
    ax = plt.axes()
    ax.plot(np.linspace(0, 10, data.shape[0]), data[:,node,sample])

def visualize_avg_sim_matrix(dm, sim_train, sim_test):
    ictal_sum = np.zeros(sim_train[0][:,:].shape)
    normal_sum = np.zeros(sim_train[0][:,:].shape)
    tc = 0
    ni = 0
    nn = 0
    for i in range(sim_train.shape[0]):
            if dm.Y_train[tc,0] == 1:
                ictal_sum = ictal_sum + sim_train[i,:,:]
                ni+=1
            else:
                normal_sum = normal_sum + sim_train[i,:,:]
                nn+=1
            tc+=1
    ictal_sum = ictal_sum / ni
    normal_sum = normal_sum / nn
    print("Average ictal")
    show_heat_map(ictal_sum)
    print("Average Non-Ictal")
    show_heat_map(normal_sum)

def visualize_sample_sim_matrix(dm, sim_train, sim_test):
    perm = np.random.permutation(sim_train.shape[0])
    train = sim_train#sim_train[perm, :, :]
    label = dm.Y_train#dm.Y_train[perm,:]

    sample_non_ictal = None
    sample_ictal = None

    for i in range(sim_train.shape[0]):
            if label[i,0] == 1 and sample_ictal is None:
                sample_ictal = train[i,:,:]

            elif label[i,0] == 0 and sample_non_ictal is None:
                sample_non_ictal = train[i,:,:]

            elif sample_ictal is not None and sample_non_ictal is not None:
                break


    print("sample ictal")
    show_heat_map(sample_ictal)
    print("sample Non-Ictal")
    show_heat_map(sample_non_ictal)