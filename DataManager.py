import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch

'''
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F

from scipy.linalg import fractional_matrix_power

import os, sys, inspect
cd = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
pd = os.path.dirname(cd)
sys.path.insert(1, pd + r'Unsupervised GCN')
from GCN import *
'''
'''
D = the input feature vector dimension
N = number of nodes
M = number of training examples

Input types
X_train: DxNxM Numpy Tensor
X_test: DxNxM Numpy Tensor
Y_train: 1xM Numpy Tensor
X_test: 1xM Numpy Tensor
n: float threshold for adjacency matrix

stored types
X_train: MxNxD Numpy Tensor
X_test: MxNxD Numpy Tensor
Y_train: Mx1 Numpy Tensor
X_test: Mx1 Numpy Tensor
A_train: NxN Numpy adjacency matrix
A_train: NxN Numpy adjacency matrix
P_avg_train: NxN Numpy covariance matrix
P_avg_test: NxN Numpy covariance matrix
'''


class dataManager:
    def __init__(self, X_train, X_test, Y_train, Y_test, n):
        # rearrange input tensors
        Y_train = np.transpose(Y_train)
        Y_test = np.transpose(Y_test)
        X_train = np.transpose(X_train, (2, 1, 0))
        X_test = np.transpose(X_test, (2, 1, 0))

        self.A_train, self.P_avg_train, indices, self.conv_avg_train = self.create_adjacency_matrix(X_train, n)
        self.train_indices = indices
        self.X_train = self.drop_samples(X_train, indices)
        self.Y_train = self.drop_samples(Y_train, indices)
        self.threshold = n

        self.A_test, self.P_avg_test, indices, self.conv_avg_test = self.create_adjacency_matrix(X_test, n)
        self.test_indices = indices
        self.X_test = self.drop_samples(X_test, indices)
        self.Y_test = self.drop_samples(Y_test, indices)
        # create adjacency matrix again using reduced X_train and X_test
        self.A_train, self.P_avg_train, indices, self.conv_avg_train = self.create_adjacency_matrix(self.X_train, n)
        self.A_test, self.P_avg_test, indices, self.conv_avg_test = self.create_adjacency_matrix(self.X_test, n)

        self.mean = None
        self.sd = None
        print("--------data manager successfully initialized--------")

    def __str__(self):
        printlist = []
        printlist.append('X_train: shape of' + str(self.X_train.shape))
        printlist.append('X_test: shape of'+ str(self.X_test.shape))
        printlist.append('Y_train: shape of'+ str(self.Y_train.shape))
        printlist.append('Y_test: shape of'+ str(self.Y_test.shape))
        printlist.append('A_train: shape of'+ str(self.A_train.shape))
        printlist.append('A_test: shape of'+ str(self.A_test.shape))
        printlist.append('P_avg_train: shape of'+ str(self.P_avg_train.shape))
        printlist.append('P_avg_test: shape of'+ str(self.P_avg_test.shape))
        return '\n'.join(printlist)
        
    def create_adjacency_matrix(self, X, n):
        '''
        Input types:
        X: MxNxD Numpy Tensor
        n: float threshold for adjacency matrix

        Returns:
        A: sparse NxN adjacency matrix with thresholding
        P_avg: raw NxN covariance matrix before thresholding
        remove_indices: indices of bad samples that should be removed
        '''
        D_0 = X.shape[2]
        M = X.shape[0]
        # X_bar is the average of each row (hence dimension M x N x 1)
        X_bar = np.mean(X, axis=2, keepdims=True)
        # buid P matrix (tensor since there are M examples) shape: MxNxN
        P = 1 / (D_0 - 1) * np.matmul((X - X_bar), np.transpose((X - X_bar), (0, 2, 1)))
        # build Aggregated P_inv averaged along M . shape: NxN
        sum_train = np.zeros(P.shape[1:])
        remove_indices = [];
        for i in range(P.shape[0]):
            # drop the samples that are non-invertable
            try:
                sum_train += np.linalg.inv(P[i, :, :])

            except:
                M -= 1;
                # print("failed at",i)
                remove_indices.append(i)
        # print(M)
        P_avg = 1 / M * sum_train
        A = (P_avg > n)
        return A, P_avg, remove_indices, np.mean(P, axis = 0)

    def drop_samples(self, X, remove_indices):
        '''
        Input types:
        X: M x * tensor where M is the number of samples
        remove_indices: list of int indices to be dropped

        Returns:
        tensor with samples from remove_indices removed
        '''
        return np.delete(X, np.s_[remove_indices], axis=0)

    def re_threshold(self, n):
        '''
        recompute adjacency matrix A using a new threshold n
        '''
        self.threshold = n
        self.A_train = (self.P_avg_train > n)
        self.A_test = (self.P_avg_test > n)

    def normalize(self):
        '''
        normalize X_train and X_test
        '''
        '''
        sample_mean = np.mean(self.X_train, axis=0, keepdims=True)
        self.mean = np.mean(sample_mean, axis=2, keepdims=True)
        self.sd = np.std(sample_mean, axis=2, keepdims=True)
        self.X_train = (self.X_train - self.mean) / self.sd
        self.X_test = (self.X_test - self.mean) / self.sd
        '''
        max = 1
        min = -1
        X = self.X_train
        X_std_train = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        self.X_train = X_std_train * (max - min) + min
        X_std_test = (self.X_test - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        self.X_test = X_std_test * (max - min) + min

    def de_normalize(self):
        '''
        unnormalize X_train and X_test
        '''
        self.X_train = self.X_train * self.sd + self.mean
        self.X_test = self.X_test * self.sd + self.mean
        self.mean = None
        self.sd = None
        
    #filter referencce: https://www.sciencedirect.com/science/article/abs/pii/S1388245711003774?via%3Dihub
    def apply_variance_filter(self, n):# n is the number of nodes to be kept
        '''
        reduce number of nodes based on variance
        '''
        ind = np.argsort(get_label_variance(self.X_train, self.Y_train))[::-1]
        #print(ind)
        ind = ind[:n]
        ind = np.sort(ind)
        self.X_train = self.X_train[:,ind,:]
        self.X_test = self.X_test[:,ind,:]
        
        # create adjacency matrix again using reduced X_train and X_test
        self.A_train, self.P_avg_train, _, self.conv_avg_train = self.create_adjacency_matrix(self.X_train, self.threshold)
        self.A_test, self.P_avg_test, _, self.conv_avg_test = self.create_adjacency_matrix(self.X_test, self.threshold)

        print("--------data successfully filtered (variance)--------")
    
    def apply_dvariance_filter(self, n):
        '''
        reduce number of nodes based on difference of variance between positive and negative samples
        '''
        ind = np.argsort(np.absolute(get_label_variance(self.X_train, self.Y_train)-get_label_variance(self.X_train, self.Y_train, 0)))[::-1]
        #print(ind)
        ind = ind[:n]
        ind = np.sort(ind)
        self.X_train = self.X_train[:,ind,:]
        self.X_test = self.X_test[:,ind,:]
        
        # create adjacency matrix again using reduced X_train and X_test
        self.A_train, self.P_avg_train, _, self.conv_avg_train = self.create_adjacency_matrix(self.X_train, self.threshold)
        self.A_test, self.P_avg_test, _, self.conv_avg_test = self.create_adjacency_matrix(self.X_test, self.threshold)

        print("--------data successfully filtered (dvariance)--------")
    
    def entropy_filter(self):
        pass
    
    def resize_clip(self, clip_size): # clip size must be divisible by the original clip size
        extend_factor = self.X_train.shape[2]//clip_size
        self.X_train = self.X_train.reshape(self.X_train.shape[0]* self.X_train.shape[2], self.X_train.shape[1])
        self.X_train = self.X_train.reshape(self.X_train.shape[0]//clip_size,self.X_train.shape[1],clip_size)
        self.X_test = self.X_test.reshape(self.X_test.shape[0]* self.X_test.shape[2], self.X_test.shape[1])
        self.X_test = self.X_test.reshape(self.X_test.shape[0]//clip_size,self.X_test.shape[1],clip_size)
        self.Y_train = np.matmul(extend_identity(self.Y_train.shape[0], extend_factor),self.Y_train)
        self.Y_test = np.matmul(extend_identity(self.Y_test.shape[0], extend_factor),self.Y_test)
        # create adjacency matrix again using resized X_train and X_test
        #self.A_train, self.P_avg_train, _ = self.create_adjacency_matrix(self.X_train, self.threshold)
        #self.A_test, self.P_avg_test, _ = self.create_adjacency_matrix(self.X_test, self.threshold)

    def mean_pooling_1d(self, size = 5, stride = 4, padding = 0):
        self.X_train = F.avg_pool1d(torch.from_numpy(self.X_train), size, stride, padding).detach().numpy()
        self.X_test = F.avg_pool1d(torch.from_numpy(self.X_test), size, stride, padding).detach().numpy()
        #self.Y_test = F.avg_pool1d(torch.from_numpy(self.Y_test), size, stride, padding).detach().numpy() > 0.5
        #self.Y_train = F.avg_pool1d(torch.from_numpy(self.Y_train), size, stride, padding).detach().numpy() > 0.5
        self.A_train, self.P_avg_train, _, self.conv_avg_train = self.create_adjacency_matrix(self.X_train, self.threshold)
        self.A_test, self.P_avg_test, _, self.conv_avg_test = self.create_adjacency_matrix(self.X_test, self.threshold)
        
def extend_identity(n, extend_factor):
    '''
    something like 
    e.g. n=4, extend factor = 3
    1
    1  
    1          0
        1
        1
        1
            1
            1
            1
      0         1
                1
                1
    '''
    identity = np.identity(n)
    ex_identity = np.zeros((n*extend_factor, n))
    for i in range(n):
        for j in range(extend_factor):
            ex_identity[i*extend_factor+j,:] = identity[i,:]
    return ex_identity
        
def get_label_variance(X,Y,label = 1):
    #X shape: MxNxD
    #Y shape: Mx1
    #output shape: Nx1
    #print(np.nonzero(Y == label))
    X = X[np.nonzero(Y == label)[0],:,:]
    #print(X.shape)
    X = np.transpose(X, (0,2,1))
    X = X.reshape(X.shape[0]*X.shape[1],X.shape[2])
    #print(X.shape)
    #print(np.mean(X, axis = 0, keepdims = True).shape)
    X = (X - np.mean(X, axis = 0, keepdims = True))**2
    return np.mean(X, axis = 0)



#-------------Archived-------------#
'''
    def sanity_check(self, t_h, t_l, parent_dir):
        
        #optional function for debugging. prints all pairs with covariance greater than t_h and less than t_l
        
        # create an averaged signal accross all X_train samples
        X_train_avg = np.mean(self.X_train, axis=0)
        X_bar_train_avg = np.mean(np.mean(self.X_train, axis=2, keepdims=True), axis=0)
        X_train_avg = np.transpose(X_train_avg) - np.transpose(X_bar_train_avg)
        # select highest covariance couples
        h_couples = []
        for i in range(self.P_avg_train.shape[0]):
            for j in range(self.P_avg_train.shape[0]):
                if i < j:
                    if self.P_avg_train[i, j] > t_h:
                        h_couples.append((i, j))
        # graph the highest covariance couples
        for couple in h_couples:
            plt.figure()
            plt.title('high covariance node ' + str(couple[0]) + " and node " + str(couple[1]))
            ax1 = plt.axes()
            ax1.plot(np.linspace(0, 10, 640), X_train_avg[:, couple[0]])
            ax1.plot(np.linspace(0, 10, 640), X_train_avg[:, couple[1]])
            plt.savefig(parent_dir + '/Covariance and Adjacency Matrix/figures/' + 'high covariance node ' + str(
                couple[0]) + " and node " + str(couple[1]) + '.png')

        # select lowest covariance couples
        l_couples = []
        for i in range(self.P_avg_train.shape[0]):
            for j in range(self.P_avg_train.shape[0]):
                if i < j:
                    if self.P_avg_train[i, j] < t_l:
                        l_couples.append((i, j))
        # graph the lowest covariance couples
        for couple in l_couples:
            plt.figure()
            plt.title('low covariance node ' + str(couple[0]) + " and node " + str(couple[1]))
            ax1 = plt.axes()
            ax1.plot(np.linspace(0, 10, 640), X_train_avg[:, couple[0]])
            ax1.plot(np.linspace(0, 10, 640), X_train_avg[:, couple[1]])
            plt.savefig(parent_dir + '/Covariance and Adjacency Matrix/figures/' + 'low covariance node ' + str(
                couple[0]) + " and node " + str(couple[1]) + '.png')
'''