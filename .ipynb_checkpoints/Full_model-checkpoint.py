from DataManager import dataManager
from DataManagerUtil import *
import GCN as GCN
import GCNutil as GCNutil
import GDE as GDE
import GDEutil as GDEutil
import GDEsolvers as GDEsolvers
import MLP as MLP

#Standard Libraries
# Torch
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
#Numpy
import numpy as np

class Integrated_Model():

    def __init__(self, GCNparams, GDEparams, MLPGCNparams, MLPGDEparams, dm, Araw):
        self.GCNparams = GCNparams
        self.GDEparams = GDEparams
        self.MLPGCNparams = MLPGCNparams
        self.MLPGDEparams = MLPGDEparams
        self.GCNmodel, self.GCNcheckpoint = self.train_GCN(dm, Araw)
        self.GDEmodel, self.GDEcheckpoint = self.train_GDE(dm, Araw)
        self.MLPmodelGCN, self.MLPmodelGDE, self.MLPcheckpointGCN, self.MLPcheckpointGDE = self.train_MLP(dm, Araw, device_name = 'cpu',acc_fn = MLP.auc2, plot_avg_matrix = True, plot_sample_matrix = True)

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

    def train_MLP(self, dm, Araw, device_name = 'cpu', acc_fn = MLP.F1, plot_avg_matrix = False, plot_sample_matrix = False):
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
        
        print("----------Training MLP-----------")
        MLPmodelGCN, _, _, MLPcheckpointGCN = MLP.train_MLP(dm, sim_all["GCN_train"], sim_all["GCN_test"],
                                                            self.MLPGCNparams, acc_fn=acc_fn, autostop_decay=0.995,
                                                            print_summary=False, verbose=False)
        print("---------------------------------")
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
        for i in range(P.shape[0]):    
            P[i, :, :] = np.linalg.inv(P[i, :, :])
        return P
    
    def approximate_covariance(self, dm, model, Araw):
        model.eval()
        A = create_DAD(Araw)
        X_train = dm.X_train
        X_test = dm.X_test
        tensor = model.get_embeddings(torch.from_numpy(X_train).float(),
                                        torch.from_numpy(A).float()).cpu().detach()
        tensor = torch.mean(model.tail.fcn(torch.cat((torch.from_numpy(X_train).float(), tensor), 2)), dim = 0)
        tensor = torch.transpose(tensor, 0, 1)
        
        sim_train = model.forward_approximate(torch.from_numpy(X_train).float(),
                                        torch.from_numpy(A).float(), tensor).cpu().detach().numpy()
        sim_test = model.forward_approximate(torch.from_numpy(X_test).float(),
                                        torch.from_numpy(A).float(), tensor).cpu().detach().numpy()
        return sim_train, sim_test
        
        
        
