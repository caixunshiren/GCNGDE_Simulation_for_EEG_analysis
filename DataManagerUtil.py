import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power
from DataManager import *

'''
DataManagerUtils are utilities functions for data visualization and helper functions.
'''



def show_heat_map(m, cmap = 'bwr'):
    '''
    shows heatmap a 2D matrix
    '''
    plt.imshow(m, cmap=cmap)
    plt.colorbar()
    plt.show()

def create_DAD(adj):
    '''
    Creates the symmetric normalization of the adjacency matrix as outlined in the GCN paper
    '''
    np.fill_diagonal(adj, 1)
    rowsum = np.sum(adj, axis=1)
    d = np.diag_indices(adj.shape[0])
    D = np.zeros(adj.shape)
    D[d] = rowsum
    #print(rowsum)
    D = fractional_matrix_power(D, -0.5)
    #print(D)
    return D@adj@D

def load_patient_data(filepath, verbose = True):
    '''
    Patient data loader to load .mat files for patient data
    Verbose: plt show example signals
    '''
    variables = {}
    f = h5py.File(filepath, 'r')
    for k, v in f.items():
        variables[k] = np.array(v)
    if verbose: 
        for key in variables.keys():
            print("name:", key)
            print("shape:", variables[key].shape)
            print("-----------------------")
        
        print('print sample EGG signal from one node:')
        plot_signal(variables["X_train"], 5,0)
    return variables

def plot_signal(data, node, sample):
    '''
    Helper function that plots a signal
    '''
    plt.figure()
    ax = plt.axes()
    ax.plot(np.linspace(0, 10, data.shape[0]), data[:,node,sample])

def visualize_avg_sim_matrix(dm, sim):
    '''
    plt show the average of a matrix of similarity matrix, differed by ictal and non-ictal state
    '''
    ictal_sum = np.zeros(sim[0][:,:].shape)
    normal_sum = np.zeros(sim[0][:,:].shape)
    ni = 0
    nn = 0
    for i in range(sim.shape[0]):
            if dm.Y_train[i,0] == 1:
                ictal_sum = ictal_sum + sim[i,:,:]
                ni+=1
            else:
                normal_sum = normal_sum + sim[i,:,:]
                nn+=1
    ictal_sum = ictal_sum / ni
    normal_sum = normal_sum / nn
    print("Average ictal")
    show_heat_map(ictal_sum)
    print("Average Non-Ictal")
    show_heat_map(normal_sum)

def visualize_sample_sim_matrix(dm, sim):
    '''
    plt show the sample similarity matrices based on ictal and non-ictal states
    '''
    train = sim
    label = dm.Y_train

    sample_non_ictal = None
    sample_ictal = None

    for i in range(sim.shape[0]):
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