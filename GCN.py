import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from crossbar import crossbar, ticket


class GCN(nn.Module):
    '''
    Graph Convolutional Neural Network model (single layer)
    Credit: Benjamin Cheng EngSci 2T2
    '''
    def __init__(self, in_features, out_features, bias=False):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ',' \
               + str(self.out_features) + ')'

    def reset_parameters(self):
        stdv = 1. / self.weight.size(1) ** 1 / 2
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    # H, feature matrix
    # A, precomputed adj matrix
    def forward(self, H, A):
        n = torch.matmul(A, torch.matmul(H, self.weight))
        if self.bias is not None:
            return n + self.bias
        else:
            return n


class SimularityMatrix(nn.Module):
    """
    Similarity matrix layer for NCDE application for GCN
    """
    def __init__(self, in_features):
        super(SimularityMatrix, self).__init__()
        self.in_features = in_features

        self.weight = nn.Parameter(torch.Tensor(in_features))
        # self.weight = torch.squeeze(self.weight)

        self.reset_parameters()

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ')'

    def reset_parameters(self):
        stdv = 1. / self.weight.size(0) ** 1 / 2
        self.weight.data.uniform_(-stdv, stdv)

    # computes the simularity matrix:
    # H, feature matrix --> N x D
    # A, precomputed adj matrix --> NxN
    def forward(self, H, H0):
        # get hidden state (concate H0 and H)
        Z = torch.cat((H0, H), 2)
        # centering normalize Z
        Z = self.fcn(Z)
        return self.get_sim(Z)

    # simularity between node u and node v (shape Mx1xD)
    # return the u,v index of the simularity matrix
    def get_sim(self, Z):  # Z is M x N Sx 2D
        theta = torch.diag(self.weight)
        return torch.matmul(torch.matmul(Z, theta), torch.transpose(Z, 1, 2))

    # centering-normalizing (CN) operator
    def fcn(self, Z):
        norm_Z = (Z - torch.mean(Z, dim=2, keepdim=True))
        return norm_Z / torch.std(Z, unbiased=True, dim=2, keepdim=True)

class Net(nn.Module):
    '''
    N-layer GCN with Similarity Matrix output layer
    '''
    def __init__(self, body_features, n_layers, activation = F.relu, bias=False):
        super(Net, self).__init__()
        assert(n_layers >= 1)
        self.activation = activation
        self.head = GCN(body_features, body_features, bias)
        self.layers = nn.ModuleList()
        for i in range(n_layers - 1):
            self.layers.append(GCN(body_features, body_features, bias))
        self.tail = SimularityMatrix(body_features*2) #SimilarityMatrixApproximate(28, body_features*2)

    def forward(self, h_0, A):
        #print(h_0.shape)
        x = self.activation(self.head(h_0, A))
        for layer in self.layers:
            x = self.activation(layer(x, A))
        sim_matrix = self.tail(x, h_0)
        return sim_matrix

    def embedding_forward(self, h_0, A):
        '''
        helper function for extracting embeddings for intermediate layer
        '''
        embeddings = []
        embeddings.append(h_0)
        x = self.activation(self.head(h_0, A))
        embeddings.append(x)
        for layer in self.layers:
            x = self.activation(layer(x, A))
            embeddings.append(x)
        sim_matrix = self.tail(x, h_0)
        return sim_matrix, embeddings

    def get_embeddings(self, h_0, A):
        '''
        helper function for getting the last embeddings before the similarity matrix layer
        '''
        x = self.activation(self.head(h_0, A))
        for layer in self.layers:
            x = self.activation(layer(x, A))
        return x


class sim_loss(torch.nn.Module):
    '''
    Customized loss function for NCDE
    '''
    def __init__(self):
        super(sim_loss, self).__init__()

    def forward(self, sim_matrix, A, epsilon=1e-8):
        A_tf = (A != 0)
        M = sim_matrix.shape[0]
        if M == 0:
            return 0
        abs_N = torch.sum(A_tf, dim=1, keepdim=True)  # Nx1 matrix

        logexp_S = torch.log(torch.sum(torch.exp(sim_matrix), dim=2, keepdim=True))

        obj_vector = (torch.sum(A_tf * sim_matrix, dim=2, keepdim=True) - abs_N * logexp_S)
        return -(1 / M) * torch.sum(obj_vector)

    

# --- CrossBar Implementation --- #
class GCN_operation(torch.autograd.Function):
    '''
    Customized autograd function for backprop with GCN on crossbar.
    '''
    @staticmethod
    def forward(ctx, ticket_A, ticket_W_T, A, W, Z):
        #Z is batched M x N x D
        ctx.save_for_backward(A, W, Z)
        Z_out = torch.zeros(Z.shape[0], A.shape[0], W.shape[1])
        for i in range(Z.shape[0]): # for each training example (M)
            # for each example, do H_i = matmul(A,Z_i), then Z_out_i = matmul(H, W) --> Z_out_i = matmul(W.T, H.T).T
            H_i = torch.zeros(A.shape[0], Z.shape[2])
            for j in range(Z.shape[2]): # for each column of Z (D)
                H_i[:,j] = torch.squeeze(ticket_A.vmm(torch.unsqueeze(Z[i,:,j],1)))
            # H_i is NxD, W is DxD_out
            Z_out_i_T = torch.zeros(Z_out.shape[2],Z_out.shape[1])
            for k in range(H_i.shape[0]): # for N nodes
                Z_out_i_T[:,k] = torch.squeeze(ticket_W_T.vmm(torch.unsqueeze(H_i[k,:],1)))
            Z_out[i, :,:] = Z_out_i_T.T
        return Z_out
        
    @staticmethod
    def backward(ctx, dZ_out):
        #worry about this later
        A, W, Z = ctx.saved_tensors #x is nxm
        dZ = torch.matmul(torch.matmul(A.T, dZ_out), W.T)
        dW = torch.matmul(torch.matmul(torch.transpose(Z, 1,2), A.T), dZ_out)
        return (None, None, None, dW, dZ)

class GCN_wCB(nn.Module):
    '''
    crossbar implemented GCN single layer
    '''
    def __init__(self, in_features, out_features, A, cb_A, cb_W):
        super(GCN_wCB, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.reset_parameters()

        #clear the crossbars
        cb_A.clear()
        cb_W.clear()


        self.A = A
        self.cb_A = cb_A
        self.cb_W_T = cb_W
        self.ticket_A = self.cb_A.register_linear(self.A) # no need to transpose since A is symmetric
        self.ticket_W_T = self.cb_W_T.register_linear(self.weight)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ',' \
               + str(self.out_features) + ')'

    def reset_parameters(self):
        stdv = 1. / self.weight.size(1) ** 1 / 2
        self.weight.data.uniform_(-stdv, stdv)

    # H, feature matrix
    # A, precomputed adj matrix
    def forward(self, Z):
        return GCN_operation.apply(self.ticket_A, self.ticket_W_T, self.A, self.weight, Z)
    
    def remap(self):
        #Should call the remap crossbar function after 1 or a couple update steps 
        self.cb_W_T.clear()
        self.ticket_W_T = self.cb_W_T.register_linear(self.weight)

class Net_wCB(nn.Module):
    '''
    Crossbar implemented N-layer GCN with similarity matrix output. i love you
    '''
    def __init__(self, body_features, n_layers, A, cb_A, cb_Ws, activation = F.relu):
        super(Net_wCB, self).__init__()
        assert(n_layers >= 1)
        self.activation = activation
        self.head = GCN_wCB(body_features, body_features, A, cb_A, cb_Ws[0])
        self.layers = nn.ModuleList()
        for i in range(n_layers - 1):
            self.layers.append(GCN_wCB(body_features, body_features, A, cb_A, cb_Ws[i+1]))
        self.tail = SimularityMatrix(body_features*2) # size(H_0 + h_u)

    def forward(self, h_0):
        #print(h_0.shape)
        x = self.activation(self.head(h_0))
        for layer in self.layers:
            x = self.activation(layer(x))
        sim_matrix = self.tail(x, h_0)
        return sim_matrix
    
    def remap(self):
        for layer in self.layers:
            if isinstance(layer, GCN_wCB):
                layer.remap()
                print("remap successfully")