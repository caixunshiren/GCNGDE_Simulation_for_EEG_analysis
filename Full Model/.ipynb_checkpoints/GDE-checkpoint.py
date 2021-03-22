import copy
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import vjp
import time
from GDEsolvers import *

# GCN Block for body layers
class Block(nn.Module):
    def __init__(self, A, features, activation, num_layers):
        super(Block, self).__init__()
        self.features = features
        self.activation = activation
        self.num_layers = num_layers
        self.A = A

    def forward(self, x, t, net_params):
        weights = net_params.view(self.num_layers, self.features, self.features)

        x = x.view(-1, self.A.size(1), self.features)
        for i in range(self.num_layers):
            x = self.A.matmul(x).matmul(weights[i, :, :])
            x = self.activation(x)

        return x

    def num_params(self):
        return self.features * self.features * self.num_layers


###=============The solver================###
# Convenience tuple -> tensor function
def flatten(*args):
    return (torch.cat(tuple(torch.flatten(arg) for arg in args), dim=0).view(1, -1))


# Convenience tensor -> tuple function
def unflatten(x, n_e, sizes):
    return (x[0, 0:n_e[0]].view(sizes[0]),
            x[0, n_e[0]:n_e[0] + n_e[1]].view(sizes[1]),
            x[0, (n_e[0] + n_e[1]):(n_e[0] + n_e[1] + n_e[2])].view(sizes[2]),
            x[0, (n_e[0] + n_e[1] + n_e[2]):].view(sizes[3]),
            )


class Integrate(torch.autograd.Function):
    def __deepcopy__(self, memo):
        return Integrate(copy.deepcopy(memo))

    @staticmethod
    def forward(ctx, Integrator, f, x0, t0, t1, N, net_params, b_tableau):
        solution = Integrator(b_tableau, lambda x, t: f(x, t, net_params), x0, t0, t1, N)

        # Save for jacobian calculations in backward()
        ctx.save_for_backward(x0, t0, t1, net_params)
        ctx.solution = solution
        ctx.Integrator = Integrator
        ctx.N = N
        ctx.f = f
        ctx.b_tableau = b_tableau

        return solution

    @staticmethod
    def backward(ctx, dL_dz1):
        # Get all saved context
        z0, t0, t1, net_params = ctx.saved_tensors
        z1 = ctx.solution
        N = ctx.N
        f = ctx.f

        # Convenience sizes
        batch_size = z0.size()[0]
        img_len = int(z0.numel() / batch_size)

        # print(dL_dz1.shape)
        # Compute derivative w.r.t. to end time of integration
        dL_dt1 = dL_dz1.reshape(batch_size, 1, -1).bmm(
            f(z1, t1, net_params).reshape(batch_size, -1, 1))  # Derivative of loss w.r.t t1

        # print("dL_dt1", dL_dt1)

        # Initial Condition
        num_elements = (z1.numel(), dL_dz1.numel(), batch_size * net_params.numel(), dL_dt1.numel())
        sizes = (z1.size(), dL_dz1.size(), (batch_size, net_params.numel()), dL_dt1.size())
        s0 = flatten(z1, dL_dz1, torch.zeros((batch_size, net_params.numel()), dtype=torch.float32, device=z1.device),
                     -dL_dt1)  # initial augmented state

        # augmented dynamics function
        def aug_dynamics(s, t, theta):
            s = unflatten(s, num_elements, sizes)

            with torch.enable_grad():
                #                 gradients = [vjp(f,
                #                                  (s[0][i].unsqueeze(0), t, theta),
                #                                   v=-s[1][i].unsqueeze(0),
                #                                  )[1] for i in range(batch_size)]
                gradients = vjp(f,
                                (s[0], t, theta),
                                v=-s[1],
                                )[1]

            return flatten(f(s[0], t, theta),
                           torch.cat([gradients[0]], dim=0),
                           torch.cat([gradients[2].unsqueeze(0) for i in range(batch_size)], dim=0),
                           torch.cat([gradients[1].reshape(1, 1) for i in range(batch_size)], dim=0),
                           )  # .unsqueeze(2)

        #             return flatten(f(s[0],t,theta),
        #                     torch.cat([gradient[0] for gradient in gradients], dim=0),
        #                     torch.cat([gradient[2].unsqueeze(0) for gradient in gradients], dim=0),
        #                     torch.cat([gradient[1].reshape(1,1) for gradient in gradients], dim=0),
        #                    )#.unsqueeze(2)

        # Integrate backwards
        with torch.enable_grad():
            s = ctx.Integrator(ctx.b_tableau, lambda x, t: aug_dynamics(x, t, net_params), s0, t1, t0, N)

        # Extract derivatives
        _, dL_dz0, dL_dtheta, dL_dt0 = unflatten(s, num_elements, sizes)

        # must return something for every input to forward, None for non-tensors
        return None, None, dL_dz0, dL_dt0, dL_dt1, None, dL_dtheta, None
###=======================================###


#ODE net
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

class ODENet(nn.Module):
    def __init__(self, solver, body_channels, hidden_layers, A, solver_params):
        super(ODENet, self).__init__()

        # Graph Laplacian
        self.A = A

        # Controls amount of parameters
        self.body_channels = body_channels
        self.f = Block(A, body_channels, F.relu, hidden_layers)

        # Head
        # self.head = GCN(A, in_channels, body_channels)

        # Body
        self.int_f = solver
        self.Integrate = Integrate
        self.solver_params = solver_params
        self.N = solver_params["N"]
        self.h = (solver_params["t1"] - solver_params["t0"]) / solver_params["N"]
        self.b_tableau = solver_params["b_tableau"]
        self.t0 = torch.tensor(float(solver_params["t0"]), requires_grad=True)
        self.t1 = torch.tensor(float(solver_params["t1"]), requires_grad=True)

        self.net_params = torch.nn.parameter.Parameter(
            torch.Tensor(self.f.num_params()).normal_(mean=0, std=0.1, generator=None), requires_grad=True)

        # Tail
        self.tail = SimularityMatrix(body_channels * 2)

    def _apply(self, fn):
        super(ODENet, self)._apply(fn)
        self.t0 = fn(self.t0)
        self.t1 = fn(self.t1)
        return self

    def forward(self, h_0):
        x = self.Integrate.apply(self.int_f, self.f, h_0, self.t0, self.t1, self.N, self.net_params, self.b_tableau)  # Vanilla RK4
        x = self.tail(x, h_0)
        return x

#Loss Function
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