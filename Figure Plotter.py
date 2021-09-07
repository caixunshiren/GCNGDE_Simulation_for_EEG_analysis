from DataManager import dataManager
from DataManagerUtil import *
import GCN as GCN
import GCNutil as GCNutil
import GDE as GDE
import GDEutil as GDEutil
import GDEsolvers as GDEsolvers
import MLP as MLP
import RNN as RNN
import Full_model as FM

#Standard Libraries
# Torch
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
#Numpy
import numpy as np

class Figure_plotter():
    '''
    This class plots and saves figures based on the Integrated_model object
    '''
    def __init__(self, patient_name, DIR):
        self.patient_name = patient_name
        self.DIR = DIR

    def sequential_plot(self,IM, dm, Araw, model_name, i=0):
        sim_all = {}
        A = create_DAD(Araw)
        X_train = dm.X_train
        X_test = dm.X_test
        device_name = 'cpu'
        # Get simularity matrix from GCN
        IM.GCNmodel.eval()
        sim_all["GCN_train"] = IM.GCNmodel(torch.from_numpy(X_train).float().to(device_name),
                                        torch.from_numpy(A).float().to(device_name)).cpu().detach().numpy()
        sim_all["GCN_test"] = IM.GCNmodel(torch.from_numpy(X_test).float().to(device_name),
                                       torch.from_numpy(A).float().to(device_name)).cpu().detach().numpy()
        # Get simularity matrix from GNODE
        IM.GDEmodel.eval()
        sim_all["GDE_train"] = IM.GDEmodel(torch.from_numpy(X_train).float().to(device_name)).cpu().detach().numpy()
        sim_all["GDE_test"] = IM.GDEmodel(torch.from_numpy(X_test).float().to(device_name)).cpu().detach().numpy()

        def save_sequential_plot(dm, sim_test, DIR, i=0):
            plt.clf()
            fig, ax = plt.subplots(nrows=2, ncols=5, constrained_layout=False)
            ictal = 0
            non_ictal = 0
            for row in ax:
                for col in row:
                    im = col.imshow(sim_test[i], cmap='bwr')
                    if dm.Y_test[i, 0] == 1:
                        ictal += 1
                        col.title.set_text("ictal {}".format(ictal))
                    else:
                        non_ictal += 1
                        col.title.set_text("non-ictal {}".format(non_ictal))
                    col.axis('off')
                    i += 1

            cbar_ax = fig.add_axes([1.1, 0.15, 0.05, 0.7])
            fig.colorbar(im, cax=cbar_ax)

            plt.savefig(DIR)
            plt.show()

        save_sequential_plot(dm, sim_all["GCN_test"], '{}/{}_{}_{}_seqential_plot'.format(self.DIR,self.patient_name,model_name,'GCN'), i)
        save_sequential_plot(dm, sim_all["GDE_test"], '{}/{}_{}_{}_seqential_plot'.format(self.DIR,self.patient_name,model_name,'GDE'), i)

    def sim_progression(self, IM, dm, Araw, model_name):
        def sequential_plot(lis, DIR):  # , DIR
            plt.clf()
            fig, ax = plt.subplots(nrows=1, ncols=len(lis), constrained_layout=True)
            # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.1)
            # plt.tight_layout()
            i = 0
            for col in ax:
                im = col.imshow(lis[i], cmap='bwr')
                col.title.set_text("H {}".format(i))

                col.axis('off')
                i += 1

            cbar_ax = fig.add_axes([1.1, 0.15, 0.05, 0.7])
            fig.colorbar(im, cax=cbar_ax)

            plt.savefig(DIR)
            plt.show()

        A = create_DAD(Araw)
        X_train = dm.X_train
        X_test = dm.X_test
        device_name = 'cpu'
        _, GCN_embeddings = IM.GCNmodel.embedding_forward(torch.from_numpy(X_test).float().to(device_name),
                                                       torch.from_numpy(A).float().to(device_name))
        for h in GCN_embeddings:
            print(h.shape)

        lis = []
        for j, h in enumerate(GCN_embeddings):
            sim_m = IM.GCNmodel.tail(h, torch.from_numpy(X_test).float().to(device_name)).cpu().detach().numpy()
            label = dm.Y_test
            sample_non_ictal = None
            sample_ictal = None

            for i in range(sim_m.shape[0]):
                if label[i, 0] == 1 and sample_ictal is None:
                    sample_ictal = sim_m[i, :, :]

                elif label[i, 0] == 0 and sample_non_ictal is None:
                    sample_non_ictal = sim_m[i, :, :]

                elif sample_ictal is not None and sample_non_ictal is not None:
                    break
            lis.append(sample_non_ictal)

        sequential_plot(lis)
        for i in range(len(GCN_embeddings) - 1):
            print("h", i + 1, "-", "h", i)
            show_heat_map(lis[i + 1] - lis[i])