import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power


#shows heatmap a 2D matrix
def show_heat_map(m):
    plt.imshow(m, cmap='hot', interpolation='nearest')
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
        
    if verbose: 
        for key in variables.keys():
            print("name:", key)
            print("shape:", variables[key].shape)
            print("-----------------------")
        
        print('print sample EGG signal from one node:')
        plt.figure()
        ax = plt.axes()
        X_train = variables["X_train"]
        X_test = variables["X_test"]
        ax.plot(np.linspace(0, 10, X_train.shape[0]), X_train[:,5,0])
    return variables