import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GCN(nn.Module):
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
    # this method is pretty wack, need to find a vectorized way to do it.
    def forward(self, H, H0):
        # get hidden state (concate H0 and H)
        Z = torch.cat((H0, H), 2)
        M = Z.shape[0]
        N = Z.shape[1]
        D = Z.shape[2]
        # centering normalize Z
        Z = self.fcn(Z)
        return self.get_sim_vectorized(Z)


    # simularity between node u and node v (shape Mx1xD)
    # return the u,v index of the simularity matrix
    def get_sim(self, u, v):
        theta = torch.diag(self.weight)
        # print(self.weight)
        # print(u.shape, theta.shape, self.weight.shape, torch.transpose(v, 1, 2).shape)
        return torch.squeeze(torch.matmul(torch.matmul(u, theta), torch.transpose(v, 1, 2)))

    def get_sim_vectorized(self, Z):  # Z is M x N Sx 2D
        theta = torch.diag(self.weight)
        sim_matrix = torch.matmul(torch.matmul(Z, theta), torch.transpose(Z, 1, 2))
        return sim_matrix

    # centering-normalizing (CN) operator
    def fcn(self, Z):
        norm_Z = (Z - torch.mean(Z, dim=2, keepdim=True))
        return norm_Z / torch.std(Z, unbiased=True, dim=2, keepdim=True)
        # (((1/(self.in_features-1)) * torch.sum(norm_Z**2, dim = 2, keepdim = True))**(1/2))

# n-layer GCN Network
class Net(nn.Module):
    def __init__(self, body_features, n_layers, activation = F.relu, bias=False):
        super(Net, self).__init__()
        assert(n_layers >= 1)
        self.activation = activation
        self.head = GCN(body_features, body_features, bias)
        self.layers = nn.ModuleList()
        for i in range(n_layers - 1):
            self.layers.append(GCN(body_features, body_features, bias))
        self.tail = SimularityMatrix(body_features*2) # size(H_0 + h_u)

    def forward(self, h_0, A):
        #print(h_0.shape)
        x = self.activation(self.head(h_0, A))
        for layer in self.layers:
            x = self.activation(layer(x, A))
        sim_matrix = self.tail(x, h_0)
        return sim_matrix


class sim_loss(torch.nn.Module):

    def __init__(self):
        super(sim_loss, self).__init__()

    def forward(self, sim_matrix, A, epsilon=1e-8):
        A_tf = (A != 0)
        M = sim_matrix.shape[0]
        abs_N = torch.sum(A_tf, dim=1, keepdim=True)  # Nx1 matrix

        logexp_S = torch.log(torch.sum(torch.exp(sim_matrix), dim=2, keepdim=True))

        obj_vector = (torch.sum(A_tf * sim_matrix, dim=2, keepdim=True) - abs_N * logexp_S)
        return -(1 / M) * torch.sum(obj_vector)


# --- CrossBar Implementation --- #
class Batched_MMM(torch.autograd.Function):
    #Modified from Louis: Custom pytorch autograd function for crossbar VMM operation
    @staticmethod
    def forward(ctx, ticket, x, W, b):
        #x shape is m x n -> convert to nxm -> the convert back
        x = torch.transpose(x,0,1)
        ctx.save_for_backward(x, W, b)
        print("debug BMMM: x is size", x.shape)
        print("debug W:", W.shape)
        print("debug b:", b.shape)
        #print(x[:,0].size, x[:,0].size(1))
        x_out = torch.zeros(W.shape[0], x.shape[1])
        for i in tqdm(range(x.shape[1])):
            #temp = ticket.vmm(torch.unsqueeze(x[:,i],1))
            #print("debug temp:", temp.shape)
            x_out[:,i] = torch.squeeze(ticket.vmm(torch.unsqueeze(x[:,i],1))) + b
        return torch.transpose(x_out,0,1)
        
    @staticmethod
    def backward(ctx, dx):
        #worry about this later
        x, W, b = ctx.saved_tensors #x is nxm
        grad_input = W.t().mm(dx)
        grad_weight = dx.mm(x.t())
        grad_bias = dx
        return (None, grad_input, grad_weight, grad_bias)

class Linear_block(nn.Module):
    '''
    Input: tensor with shape m x n x _
    Weight: matrix of shape _ x n
    Output: tensor with shape m x _ x _
    
    in_size: n
    out_size: _
    
    '''
    def __init__(self, in_size, out_size, cb_param, w = None, b = None):
        super(Linear_block, self).__init__()
        if w is not None and b is not None:
            self.w = nn.Parameter(w)
            self.b = nn.Parameter(b)
            print("--- weight initialized successfually ---")
        else:
            stdv = 1. / in_size ** 1 / 2
            self.w = nn.Parameter(torch.Tensor(out_size, in_size)).data.uniform_(-stdv, stdv)
            self.b = nn.Parameter(torch.Tensor(out_size, 1)).data.uniform_(-stdv, stdv)
        self.cb = crossbar(cb_param)
        self.f = Batched_VMM()
        #print("debug:",self.w.shape)
        self.ticket = self.cb.register_linear(torch.transpose(self.w, 0,1))
        
    def forward(self, x):
        return self.f.apply(self.ticket, x, self.w, self.b)
    
    def remap(self):
        #Should call the remap crossbar function after 1 or a couple update steps 
        self.cb.clear()
        self.ticket = self.cb.register_linear(torch.transpose(self.w, 0,1))