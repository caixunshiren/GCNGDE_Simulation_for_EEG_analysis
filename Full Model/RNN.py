import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import copy
from crossbar import crossbar, ticket
from tqdm import tqdm

class Simple_block(nn.module):
    def __init__(self, in_dim, out_dim, hidden_dim, activation_h = nn.Tanh(), activation_o = nn.Sigmoid()):
        #h+1 = activation(Win*input + Wh*hidden + bh)
        #out = activation(Wout*h+1 + bo)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.activation_h = activation_h
        self.activation_o = activation_o

        self.Win = nn.Parameter(torch.Tensor(in_dim, hidden_dim))
        self.bh = nn.Parameter(torch.Tensor(1, hidden_dim))
        self.Wh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.Wout = nn.Parameter(torch.Tensor(hidden_dim, out_dim))
        self.bo = nn.Parameter(torch.Tensor(1, hidden_dim))

    def reset_parameters(self):
        stdv = 1. / self.Win.size(1) ** 1 / 2
        self.Win.data.uniform_(-stdv, stdv)

        stdv = 1. / self.bh.size(1) ** 1 / 2
        self.bh.data.uniform_(-stdv, stdv)

        stdv = 1. / self.Wh.size(1) ** 1 / 2
        self.Wh.data.uniform_(-stdv, stdv)

        stdv = 1. / self.Wout.size(1) ** 1 / 2
        self.Wout.data.uniform_(-stdv, stdv)

        stdv = 1. / self.bo.size(1) ** 1 / 2
        self.bo.data.uniform_(-stdv, stdv)

    def forward(self, X, h):
        #input tensors are in the form m x n where m is the number training examples and n is the feature dimension
        #compute hidden state
        h_1 = self.activation_h(torch.matmul(X, self.Win) + torch.matmul(h, self.Wh) + self.bh)
        out = self.activation_o(torch.matmul(h_1, self.Wout) + self.bo)

        return out, h_1


class RNN(nn.module):
    def __init__(self, in_dim, out_dim, hidden_dim, activation_h = nn.Tanh(), activation_o = nn.Sigmoid()):
        self.block = Simple_block(in_dim, out_dim, hidden_dim, activation_h, activation_o)
        




























