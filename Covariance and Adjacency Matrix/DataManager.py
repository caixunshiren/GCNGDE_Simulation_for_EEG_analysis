import numpy as np
import matplotlib.pyplot as plt
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

        self.A_train, self.P_avg_train, indices = self.create_adjacency_matrix(X_train, n)
        self.X_train = self.drop_samples(X_train, indices)
        self.Y_train = self.drop_samples(Y_train, indices)

        self.A_test, self.P_avg_test, indices = self.create_adjacency_matrix(X_test, n)
        self.X_test = self.drop_samples(X_test, indices)
        self.Y_test = self.drop_samples(Y_test, indices)
        #create adjacency matrix again using reduced X_train and X_test
        self.A_train, self.P_avg_train, indices = self.create_adjacency_matrix(self.X_train, n)
        self.A_test, self.P_avg_test, indices = self.create_adjacency_matrix(self.X_test, n)

        self.mean = None
        self.sd = None
        print("--------data manager successfully initialized--------")

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
                #print("failed at",i)
                remove_indices.append(i)
        #print(M)
        P_avg = 1 / M * sum_train
        A = (P_avg > n)
        return A, P_avg, remove_indices

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
        self.A_train = (self.P_avg_train > n)
        self.A_test = (self.P_avg_test > n)

    def normalize(self):
        '''
        normalize X_train and X_test
        '''
        sample_mean = np.mean(self.X_train, axis=0, keepdims=True)
        self.mean = np.mean(sample_mean, axis=2, keepdims=True)
        self.sd = np.std(sample_mean, axis=2, keepdims=True)
        self.X_train = (self.X_train - self.mean) / self.sd
        self.X_test = (self.X_test - self.mean) / self.sd

    def de_normalize(self):
        '''
        unnormalize X_train and X_test
        '''
        self.X_train = self.X_train * self.sd + self.mean
        self.X_test = self.X_test * self.sd + self.mean
        self.mean = None
        self.sd = None

    def sanity_check(self, t_h, t_l, parent_dir):
        '''
        optional function for debugging. prints all pairs with covariance greater than t_h and less than t_l
        '''
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
            ax1.plot(np.linspace(0, 10, 640), X_train_avg[:, couple[0]]);
            ax1.plot(np.linspace(0, 10, 640), X_train_avg[:, couple[1]]);
            plt.savefig(parent_dir+'/Covariance and Adjacency Matrix/figures/'+'high covariance node ' + str(couple[0]) + " and node " + str(couple[1])+'.png')

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
            ax1.plot(np.linspace(0, 10, 640), X_train_avg[:, couple[0]]);
            ax1.plot(np.linspace(0, 10, 640), X_train_avg[:, couple[1]]);
            plt.savefig(parent_dir+'/Covariance and Adjacency Matrix/figures/' + 'low covariance node ' + str(couple[0]) + " and node " + str(couple[1]) + '.png')
#debug code
'''
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
import h5py
import numpy as np
filepath = parent_dir+'\datasets\sample_patients\pat_FR_620.mat'
variables = {}
f = h5py.File(filepath, 'r')
for k, v in f.items():
    variables[k] = np.array(v)
for key in variables.keys():
    print("name:", key)
    print("shape:", variables[key].shape)
    print("-----------------------")
#print sample EGG signal from one node
import matplotlib.pyplot as plt

plt.figure()
ax = plt.axes()
X_train = variables["X_train"]
X_test = variables["X_test"]

ax.plot(np.linspace(0, 10, 640), X_train[:,5,0])
plt.savefig(parent_dir+'/Covariance and Adjacency Matrix/figures/sample_node.png')
dm = dataManager(variables["X_train"],variables["X_test"],variables["y_train"],variables["y_test"],10**(-4))

#dm.sanity_check(4.514*10**-1, -10**-3)
#print(dm.P_avg_train)
print(dm.X_train, dm.X_train.shape)
print('----------------------------------------------------')
dm.normalize()
print(dm.X_train, dm.X_train.shape)
print('----------------------------------------------------')
dm.de_normalize()
print(dm.X_train, dm.X_train.shape)
'''