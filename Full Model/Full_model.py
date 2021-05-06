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
        self.MLPmodelGCN, self.MLPmodelGDE, self.MLPcheckpointGCN, self.MLPcheckpointGDE = self.train_MLP(dm, Araw)

    def train_GCN(self, dm, Araw, device_name = 'cpu'):
        A = create_DAD(Araw)
        X_train = dm.X_train
        X_test = dm.X_test
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

    def train_MLP(self, dm, Araw, device_name = 'cpu', acc_fn = MLP.F1):
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

        print("----------Training MLP-----------")
        MLPmodelGCN, _, _, MLPcheckpointGCN = MLP.train_MLP(dm, sim_all["GCN_train"], sim_all["GCN_test"],
                                                            self.MLPGCNparams, acc_fn=acc_fn, autostop_decay=0.995,
                                                            print_summary=False, verbose=False)

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
        print("GDE accuracy:")
        for i in range(5, 100, 5):
            t = i / 100
            MLP.eval_mlp(self.MLPmodelGDE, sim_all["GDE_test"], dm, device_name='cuda', threshold=t)