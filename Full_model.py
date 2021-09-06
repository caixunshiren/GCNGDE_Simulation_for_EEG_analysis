from DataManager import dataManager
from DataManagerUtil import *
import GCN as GCN
import GCNutil as GCNutil
import GDE as GDE
import GDEutil as GDEutil
import GDEsolvers as GDEsolvers
import MLP as MLP
import RNN as RNN

#Standard Libraries
# Torch
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
#Numpy
import numpy as np

'''
Integrated model that builds a GCN, GDE -> MLP, RNN pipeline
'''
class Integrated_Model():

    def __init__(self, GCNparams, GDEparams, MLPGCNparams, MLPGDEparams, dm, Araw):
        self.GCNparams = GCNparams
        self.GDEparams = GDEparams
        self.MLPGCNparams = MLPGCNparams
        self.MLPGDEparams = MLPGDEparams
        self.GCNmodel, self.GCNcheckpoint = self.train_GCN(dm, Araw)
        self.GDEmodel, self.GDEcheckpoint = self.train_GDE(dm, Araw)
        self.MLPmodelGCN, self.MLPmodelGDE, self.MLPcheckpointGCN, self.MLPcheckpointGDE = self.train_MLP(dm, Araw, acc_fn = MLP.auc, plot_avg_matrix = True, plot_sample_matrix = True)

    def train_GCN(self, dm, Araw, device_name = 'cpu'):
        A = create_DAD(Araw)
        X_train = dm.X_train
        X_test = dm.X_test
        #print(X_train.shape, X_test.shape, A.shape)
        GCNcheckpoint = {'parameters': self.GCNparams}
        print("----------Training GCN-----------")
        GCNmodel, GCNcheckpoint = GCNutil.train_GCN(A, X_train, X_test, GCNcheckpoint, device_name=device_name,
                                                    load=False, print_summary=False)
        print("----------Training Ends-----------")
        return GCNmodel, GCNcheckpoint

    def train_GDE(self, dm, Araw, device_name = 'cpu'):
        A = create_DAD(Araw)
        X_train = dm.X_train
        X_test = dm.X_test
        GDEcheckpoint = {'parameters': self.GDEparams}
        print("----------Training GDE-----------")
        GDEmodel, GDEcheckpoint = GDEutil.train_GDE(A, X_train, X_test, GDEcheckpoint, device_name=device_name,
                                                    load=False, print_summary=False)
        print("----------Training Ends-----------")
        return GDEmodel, GDEcheckpoint

    def train_MLP(self, dm, Araw, acc_fn = MLP.F1, plot_avg_matrix = False, plot_sample_matrix = False):
        device_name = 'cpu'
        sim_all = {}
        A = create_DAD(Araw)
        X_train = dm.X_train
        X_test = dm.X_test
        GCNmodel = self.GCNmodel
        GDEmodel = self.GDEmodel

        # Get simularity matrix from GCN
        GCNmodel.eval()
        
        sim_all["GCN_train"] = GCNmodel(torch.from_numpy(X_train).float().to(device_name),
                                        torch.from_numpy(A).float().to(device_name)).cpu().detach().numpy()
        sim_all["GCN_test"] = GCNmodel(torch.from_numpy(X_test).float().to(device_name),
                                       torch.from_numpy(A).float().to(device_name)).cpu().detach().numpy()
        
        # Get simularity matrix from GNODE
        GDEmodel.eval()
        sim_all["GDE_train"] = GDEmodel(torch.from_numpy(X_train).float().to(device_name)).cpu().detach().numpy()
        sim_all["GDE_test"] = GDEmodel(torch.from_numpy(X_test).float().to(device_name)).cpu().detach().numpy()

        if plot_avg_matrix:
            print("Average Ictal and Non-Ictal Simularity Matrix for GCN")
            visualize_avg_sim_matrix(dm, sim_all["GCN_train"])
            print("Average Ictal and Non-Ictal Simularity Matrix for GDE")
            visualize_avg_sim_matrix(dm, sim_all["GDE_train"])
            
        if plot_sample_matrix:
            print("Sample Ictal and Non-Ictal Simularity Matrix for GCN")
            visualize_sample_sim_matrix(dm, sim_all["GCN_train"])
            print("Sample Ictal and Non-Ictal Simularity Matrix for GDE")
            visualize_sample_sim_matrix(dm, sim_all["GDE_train"])
        
        print("----------Training MLP-----------")
        MLPmodelGCN, _, _, MLPcheckpointGCN = MLP.train_MLP(dm, sim_all["GCN_train"], sim_all["GCN_test"],
                                                            self.MLPGCNparams, acc_fn=acc_fn, autostop_decay=0.995,
                                                            print_summary=False, verbose=False)
        print("----------------------------------")
        MLPmodelGDE, _, _, MLPcheckpointGDE = MLP.train_MLP(dm, sim_all["GDE_train"], sim_all["GDE_test"],
                                                            self.MLPGDEparams, acc_fn=acc_fn, autostop_decay=0.995,
                                                            print_summary=False, verbose=False)
        print("----------Training Ends-----------")

        return MLPmodelGCN, MLPmodelGDE, MLPcheckpointGCN, MLPcheckpointGDE

    def print_accuracy(self, dm, Araw, device_name = 'cpu'):
        sim_all = {}
        A = create_DAD(Araw)
        X_test = dm.X_test
        GCNmodel = self.GCNmodel
        GDEmodel = self.GDEmodel

        # Get simularity matrix from GCN
        GCNmodel.eval()
        sim_all["GCN_test"] = GCNmodel(torch.from_numpy(X_test).float().to(device_name),
                                       torch.from_numpy(A).float().to(device_name)).cpu().detach().numpy()

        # Get simularity matrix from GNODE
        GDEmodel.eval()
        sim_all["GDE_test"] = GDEmodel(torch.from_numpy(X_test).float().to(device_name)).cpu().detach().numpy()

        print("GCN accuracy:")
        for i in range(5, 100, 5):
            t = i / 100
            MLP.eval_mlp(self.MLPmodelGCN, sim_all["GCN_test"], dm, device_name='cuda', threshold=t)
        MLP.eval_plot_MLP(self.MLPmodelGCN, sim_all["GCN_test"], dm, device_name='cuda')
        
        print("GDE accuracy:")
        for i in range(5, 100, 5):
            t = i / 100
            MLP.eval_mlp(self.MLPmodelGDE, sim_all["GDE_test"], dm, device_name='cuda', threshold=t)
        MLP.eval_plot_MLP(self.MLPmodelGDE, sim_all["GDE_test"], dm, device_name='cuda')
        
    def inv_convariance(self, X):
        D_0 = X.shape[2]
        M = X.shape[0]
        # X_bar is the average of each row (hence dimension M x N x 1)
        X_bar = np.mean(X, axis=2, keepdims=True)
        # buid P matrix (tensor since there are M examples) shape: MxNxN
        P = 1 / (D_0 - 1) * np.matmul((X - X_bar), np.transpose((X - X_bar), (0, 2, 1)))
        for i in range(M):
            P[i, :, :] = np.linalg.inv(P[i, :, :])
        return P


class Integrated_Model_RNN():

    def __init__(self, GCNparams, GDEparams, RNNGCNparams, RNNGDEparams, dm, Araw):
        self.GCNparams = GCNparams
        self.GDEparams = GDEparams
        self.RNNGCNparams = RNNGCNparams
        self.RNNGDEparams = RNNGDEparams
        self.GCNmodel, self.GCNcheckpoint = self.train_GCN(dm, Araw)
        self.GDEmodel, self.GDEcheckpoint = self.train_GDE(dm, Araw)
        self.RNNmodelGCN, self.RNNmodelGDE, self.RNNcheckpointGCN, self.RNNcheckpointGDE = self.train_RNN(dm, Araw,
                                                                                                          device_name='cpu',
                                                                                                          acc_fn=MLP.auc,
                                                                                                          plot_avg_matrix=False,
                                                                                                          plot_sample_matrix=False)

    def train_GCN(self, dm, Araw, device_name='cpu'):
        A = create_DAD(Araw)
        X_train = dm.X_train
        X_test = dm.X_test
        # print(X_train.shape, X_test.shape, A.shape)
        GCNcheckpoint = {'parameters': self.GCNparams}
        print("----------Training GCN-----------")
        GCNmodel, GCNcheckpoint = GCNutil.train_GCN(A, X_train, X_test, GCNcheckpoint, device_name=device_name,
                                                    load=False, print_summary=False)
        print("----------Training Ends-----------")
        return GCNmodel, GCNcheckpoint

    def train_GDE(self, dm, Araw, device_name='cpu'):
        A = create_DAD(Araw)
        X_train = dm.X_train
        X_test = dm.X_test
        GDEcheckpoint = {'parameters': self.GDEparams}
        print("----------Training GDE-----------")
        GDEmodel, GDEcheckpoint = GDEutil.train_GDE(A, X_train, X_test, GDEcheckpoint, device_name=device_name,
                                                    load=False, print_summary=False)
        print("----------Training Ends-----------")
        return GDEmodel, GDEcheckpoint

    def train_RNN(self, dm, Araw, device_name='cpu', acc_fn=MLP.F1, plot_avg_matrix=False, plot_sample_matrix=False):
        sim_all = {}
        A = create_DAD(Araw)
        X_train = dm.X_train
        X_test = dm.X_test
        GCNmodel = self.GCNmodel
        GDEmodel = self.GDEmodel

        # Get simularity matrix from GCN
        GCNmodel.eval()

        sim_all["GCN_train"] = GCNmodel(torch.from_numpy(X_train).float().to(device_name),
                                        torch.from_numpy(A).float().to(device_name)).cpu().detach().numpy()
        sim_all["GCN_test"] = GCNmodel(torch.from_numpy(X_test).float().to(device_name),
                                       torch.from_numpy(A).float().to(device_name)).cpu().detach().numpy()
        '''
        sim_all["GCN_train"], sim_all["GCN_test"]= self.approximate_covariance(dm, GCNmodel, Araw)
        '''

        # Get simularity matrix from GNODE
        GDEmodel.eval()
        sim_all["GDE_train"] = GDEmodel(torch.from_numpy(X_train).float().to(device_name)).cpu().detach().numpy()
        sim_all["GDE_test"] = GDEmodel(torch.from_numpy(X_test).float().to(device_name)).cpu().detach().numpy()

        if plot_avg_matrix:
            print("Average Ictal and Non-Ictal Simularity Matrix for GCN")
            visualize_avg_sim_matrix(dm, sim_all["GCN_train"], sim_all["GCN_test"])
            print("Average Ictal and Non-Ictal Simularity Matrix for GDE")
            visualize_avg_sim_matrix(dm, sim_all["GDE_train"], sim_all["GDE_test"])

        if plot_sample_matrix:
            print("Sample Ictal and Non-Ictal Simularity Matrix for GCN")
            visualize_sample_sim_matrix(dm, sim_all["GCN_train"], sim_all["GCN_test"])
            print("Sample Ictal and Non-Ictal Simularity Matrix for GDE")
            visualize_sample_sim_matrix(dm, sim_all["GDE_train"], sim_all["GDE_test"])

        print("----------Training RNN-----------")
        #train_RNN(dm, sim_train, sim_test, parameters, acc_fn= F1, autostop_decay=0.995, print_summary=True, verbose=True)
        RNNmodelGCN, _, _, RNNcheckpointGCN = RNN.train_RNN(dm, sim_all["GCN_train"], sim_all["GCN_test"],
                                                            self.RNNGCNparams, acc_fn=acc_fn, autostop_decay=0.995,
                                                            print_summary=False, verbose=True)
        print("----------------------------------")
        RNNmodelGDE, _, _, RNNcheckpointGDE = RNN.train_RNN(dm, sim_all["GDE_train"], sim_all["GDE_test"],
                                                            self.RNNGDEparams, acc_fn=acc_fn, autostop_decay=0.995,
                                                            print_summary=False, verbose=True)
        print("----------Training Ends-----------")

        return RNNmodelGCN, RNNmodelGDE, RNNcheckpointGCN, RNNcheckpointGDE

    def print_accuracy(self, dm, Araw, device_name='cpu'):
        sim_all = {}
        A = create_DAD(Araw)
        X_test = dm.X_test
        GCNmodel = self.GCNmodel
        GDEmodel = self.GDEmodel

        # Get simularity matrix from GCN
        GCNmodel.eval()
        sim_all["GCN_test"] = GCNmodel(torch.from_numpy(X_test).float().to(device_name),
                                       torch.from_numpy(A).float().to(device_name)).cpu().detach().numpy()

        # Get simularity matrix from GNODE
        GDEmodel.eval()
        sim_all["GDE_test"] = GDEmodel(torch.from_numpy(X_test).float().to(device_name)).cpu().detach().numpy()

        print("GCN accuracy:")
        for i in range(5, 100, 5):
            t = i / 100
            RNN.eval_RNN(self.RNNmodelGCN, sim_all["GCN_test"], dm, device_name='cuda', threshold=t)
        RNN.eval_plot_RNN(self.RNNmodelGCN, sim_all["GCN_test"], dm, device_name='cuda')

        print("GDE accuracy:")
        for i in range(5, 100, 5):
            t = i / 100
            RNN.eval_RNN(self.RNNmodelGDE, sim_all["GDE_test"], dm, device_name='cuda', threshold=t)
        RNN.eval_plot_RNN(self.RNNmodelGDE, sim_all["GDE_test"], dm, device_name='cuda')