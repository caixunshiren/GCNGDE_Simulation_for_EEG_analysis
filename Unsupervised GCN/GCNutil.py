import numpy as np
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import fractional_matrix_power
import matplotlib.pyplot as plt
from GCN import *
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(1, parent_dir+r'\Covariance and Adjacency Matrix')
from DataManager import dataManager



def show_heat_map(m):
    plt.imshow(m, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.show()

def load_ckp(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    return checkpoint

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


def load_ckp(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    return checkpoint


# get sim matrix from unsupervised model
def get_sim_matrix_from_model(dm, model_dir):
    checkpoint = load_ckp(model_dir)

    # preprocess inputs
    A = torch.from_numpy(create_DAD(dm.A_train)).float()
    X_train = torch.from_numpy(dm.X_train).float()
    X_test = torch.from_numpy(dm.X_test).float()
    device = torch.device('cpu')
    A = A.to(device)
    parameters = checkpoint['parameters']
    input_train = X_train.to(device)
    input_test = X_test.to(device)

    # initialize model
    model = Net(parameters['body'], parameters['n_layers'], F.relu, bias=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters['learning_rate'],
                                 weight_decay=parameters['weight_decay'])

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    model.eval()
    sim_train = []
    sim_test = []
    permutation = range(input_train.shape[0])
    batch_size = parameters['batch_size']

    sim1 = model(input_train, A).to(device)
    sim2 = model(input_test, A).to(device)
    return sim1.cpu().detach().numpy(), sim2.cpu().detach().numpy()


# get pretrained sim matrix
def load_flattened_sim_matrix(DIR, dm):
    sim_train = np.load(DIR + r"/pat_FR_620_Training_Similarities.npy")
    sim_test = np.load(DIR + r"/pat_FR_620_Testing_Similarities.npy")

    sim_train = np.delete(sim_train, np.s_[dm.train_indices], axis=0)
    sim_test = np.delete(sim_test, np.s_[dm.test_indices], axis=0)

    sim_train_list = []
    tri_indices = torch.triu_indices(31, 31)
    for j in range(sim_train.shape[0]):
        sim = np.zeros((1, 31, 31))
        for i in range(496):
            sim[0, int(tri_indices[0, i]), int(tri_indices[1, i])] = sim_train[j, i]
        sim_train_list.append(sim)

    sim_test_list = []
    for j in range(sim_test.shape[0]):
        sim = np.zeros((1, 31, 31))
        for i in range(496):
            sim[0, int(tri_indices[0, i]), int(tri_indices[1, i])] = sim_test[j, i]
        sim_test_list.append(sim)

    return np.vstack(sim_train_list), np.vstack(sim_test_list)


